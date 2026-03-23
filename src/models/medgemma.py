# -*- coding: utf-8 -*-
"""
src/models/medgemma.py
======================
On-demand clinical report generation via MedGemma.

Classes
-------
MedGemmaService
    Loads MedGemma once into GPU memory and serves report requests.
    Does NOT run during training — invoked only when a clinician requests
    a report, or via the REST API.

MedGemmaAPI
    Wraps MedGemmaService as an HTTP REST server.
    Tries Flask first; falls back to Python stdlib ``http.server``.

Doctor calls the API
--------------------
    POST http://HOST:8787/report
        image=<file>  pred_class=MEL  pred_conf=0.87
        box_cx=0.52   box_cy=0.48    box_w=0.31  box_h=0.28

    GET  http://HOST:8787/health
"""

import io
import json
import time
import traceback
from typing import Dict, Tuple

import torch
from PIL import Image


# ==============================================================================
# CLINICAL PROMPT
# ==============================================================================

_CLINICAL_PROMPT = (
    "You are a warm, expert clinical decision support assistant specialised in dermoscopy.\n"
    "An AI model has analysed a skin lesion image and produced the following result:\n\n"
    "  Predicted lesion type : {pred_class}\n"
    "  Model confidence      : {conf_pct:.1f}%\n"
    "  Lesion location       : x1={x1}px  y1={y1}px  x2={x2}px  y2={y2}px "
    "(image {W}x{H}px)\n"
    "  Lesion covers         : {size_pct:.1f}% of the image area\n\n"
    "Write a clinical report as flowing, natural paragraphs — no numbered lists, "
    "no bullet points, no section headers. The report must be understandable to "
    "BOTH the treating dermatologist AND the patient.\n\n"
    "Your response should read like a letter from a specialist. Follow this flow:\n\n"
    "Paragraph 1 — What was found: Describe what the AI detected and where in the "
    "image the lesion is located. Use plain English the patient can understand.\n\n"
    "Paragraph 2 — What it looks like: Describe the visible features of the lesion "
    "that support the diagnosis — shape, border, colour variation, size. "
    "Mention relevant dermoscopic criteria (e.g. ABCD rule) naturally in the text.\n\n"
    "Paragraph 3 — What this means for risk: State the risk level clearly "
    "(e.g. 'This finding carries a high level of clinical concern') and explain "
    "why in one or two sentences a patient can follow.\n\n"
    "Paragraph 4 — What should happen next: Recommend the immediate next clinical "
    "step (monitoring, biopsy, referral, urgent oncology consult). Be specific.\n\n"
    "Paragraph 5 — A brief, honest reminder: In one gentle sentence, note that "
    "this report was produced by an AI and must be confirmed by a qualified "
    "dermatologist before any clinical action is taken.\n\n"
    "Tone: professional, warm, reassuring where the risk is low, honest and "
    "urgent where the risk is high. Do not use jargon without explaining it. "
    "Maximum 280 words total."
)


# ==============================================================================
# MEDGEMMA SERVICE
# ==============================================================================

class MedGemmaService:
    """
    Loads MedGemma once and serves on-demand clinical report requests.

    MedGemma is lazy-loaded on the first :meth:`generate_report` call
    and stays resident in GPU memory until :meth:`unload` is called.

    Parameters
    ----------
    model_id         : str  — HuggingFace model ID or local path
    cache_dir        : str  — HuggingFace cache directory
    use_4bit         : bool — load in 4-bit quantisation (default True)
    local_files_only : bool — never fetch from HuggingFace Hub (default True)
    max_new_tokens   : int  — max tokens in generated report

    Example
    -------
    >>> svc = MedGemmaService(model_id, cache_dir)
    >>> result = svc.generate_report(
    ...     image_path="patient.jpg",
    ...     pred_class="MEL", pred_conf=0.87,
    ...     box_cx=0.52, box_cy=0.48, box_w=0.31, box_h=0.28)
    >>> print(result["report"])
    """

    def __init__(
        self,
        model_id:         str,
        cache_dir:        str,
        use_4bit:         bool = True,
        local_files_only: bool = True,
        max_new_tokens:   int  = 400,
    ):
        self._model_id         = model_id
        self._cache_dir        = cache_dir
        self._use_4bit         = use_4bit
        self._local_files_only = local_files_only
        self.max_new_tokens    = max_new_tokens
        self._model            = None
        self._processor        = None
        self._loaded           = False

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def generate_report(
        self,
        image_path,
        pred_class: str,
        pred_conf:  float,
        box_cx:     float,
        box_cy:     float,
        box_w:      float,
        box_h:      float,
    ) -> Dict:
        """
        Generate a structured clinical report for a single detection.

        Parameters
        ----------
        image_path : str, Path, or raw bytes
        pred_class : class name from YOLO (e.g. ``"MEL"``)
        pred_conf  : confidence score in [0, 1]
        box_cx/cy  : normalised box centre
        box_w/h    : normalised box dimensions

        Returns
        -------
        dict with keys:
            ``report``, ``pred_class``, ``pred_conf``,
            ``box_pixels``, ``image_size``, ``gen_time_sec``
        """
        self._load()

        # Load image
        if isinstance(image_path, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_path)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")
        W, H = img.size

        # Convert normalised box → pixel coords
        x1 = int((box_cx - box_w / 2) * W)
        y1 = int((box_cy - box_h / 2) * H)
        x2 = int((box_cx + box_w / 2) * W)
        y2 = int((box_cy + box_h / 2) * H)
        size_pct = box_w * box_h * 100.0

        prompt_text = _CLINICAL_PROMPT.format(
            pred_class=pred_class,
            conf_pct=pred_conf * 100,
            x1=x1, y1=y1, x2=x2, y2=y2,
            W=W, H=H, size_pct=size_pct,
        )

        # Build model inputs
        try:
            msgs   = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}]
            prompt = self._processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=prompt, images=[img], return_tensors="pt")
        except Exception:
            inputs = self._processor(
                text=prompt_text, images=[img], return_tensors="pt")

        dev    = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen_time = time.time() - t0

        decoded = self._processor.batch_decode(
            out, skip_special_tokens=True)[0]
        # Strip any echoed prompt from the response
        for marker in [prompt_text[:60], "assistant\n", "model\n"]:
            idx = decoded.find(marker)
            if idx != -1:
                decoded = decoded[idx + len(marker):]

        return {
            "pred_class":    pred_class,
            "pred_conf":     round(pred_conf, 4),
            "box_pixels":    {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_size":    {"W": W, "H": H},
            "report":        decoded.strip(),
            "gen_time_sec":  round(gen_time, 2),
        }

    def unload(self) -> None:
        """Free GPU memory when the service is no longer needed."""
        self._model     = None
        self._processor = None
        self._loaded    = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[medgemma] Model unloaded.")

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the model — stays in GPU memory once loaded."""
        if self._loaded:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[medgemma] Loading: {self._model_id}")
        t0 = time.time()

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kw    = dict(
            device_map="auto",
            torch_dtype=dtype,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )
        if self._use_4bit:
            kw["load_in_4bit"] = True

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_id, **kw)
        except TypeError:
            kw.pop("load_in_4bit", None)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_id, **kw)

        self._loaded = True
        print(f"[medgemma] Loaded in {time.time() - t0:.1f}s")

    def __repr__(self) -> str:
        return (
            f"MedGemmaService("
            f"model_id={self._model_id!r}, "
            f"loaded={self._loaded}, "
            f"4bit={self._use_4bit})"
        )


# ==============================================================================
# MEDGEMMA API SERVER
# ==============================================================================

class MedGemmaAPI:
    """
    HTTP REST server that exposes :class:`MedGemmaService` as an endpoint.

    MedGemma is lazy-loaded on the first ``/report`` request.

    Tries Flask first; falls back to Python stdlib ``http.server``.

    Endpoints
    ---------
    POST /report  — multipart: ``image`` file + form fields
    GET  /health  — returns model load status

    Parameters
    ----------
    service : MedGemmaService
    host    : str  (default ``"0.0.0.0"``)
    port    : int  (default ``8787``)

    Example
    -------
    >>> api = MedGemmaAPI(service, port=8787)
    >>> api.serve()   # blocks until killed
    """

    def __init__(
        self,
        service: MedGemmaService,
        host:    str = "0.0.0.0",
        port:    int = 8787,
    ):
        self.service = service
        self.host    = host
        self.port    = port

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def serve(self) -> None:
        """Start the server. Blocks until killed."""
        try:
            self._serve_flask()
        except ImportError:
            print("[api] Flask not installed — using stdlib HTTP server")
            self._serve_stdlib()

    # ------------------------------------------------------------------
    # PRIVATE — routing
    # ------------------------------------------------------------------

    def _handle_report(
        self, image_bytes: bytes, form: dict
    ) -> Tuple[int, dict]:
        try:
            result = self.service.generate_report(
                image_path = image_bytes,
                pred_class = form.get("pred_class", "UNKNOWN"),
                pred_conf  = float(form.get("pred_conf", 0.5)),
                box_cx     = float(form.get("box_cx",   0.5)),
                box_cy     = float(form.get("box_cy",   0.5)),
                box_w      = float(form.get("box_w",    0.4)),
                box_h      = float(form.get("box_h",    0.4)),
            )
            return 200, result
        except Exception as e:
            traceback.print_exc()
            return 500, {"error": str(e)}

    # ------------------------------------------------------------------
    # PRIVATE — Flask backend
    # ------------------------------------------------------------------

    def _serve_flask(self) -> None:
        from flask import Flask, jsonify, request

        app = Flask("medgemma_api")

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "ok",
                "model":  self.service._model_id,
                "loaded": self.service._loaded,
            })

        @app.route("/report", methods=["POST"])
        def report():
            if "image" not in request.files:
                return jsonify({"error": "Missing 'image' file field"}), 400
            image_bytes = request.files["image"].read()
            form        = {k: v for k, v in request.form.items()}
            status, res = self._handle_report(image_bytes, form)
            return jsonify(res), status

        print(f"[api] Flask server: http://{self.host}:{self.port}")
        print("[api] POST /report  |  GET /health")
        app.run(host=self.host, port=self.port, debug=False, threaded=False)

    # ------------------------------------------------------------------
    # PRIVATE — stdlib fallback
    # ------------------------------------------------------------------

    def _serve_stdlib(self) -> None:
        import cgi
        from http.server import BaseHTTPRequestHandler, HTTPServer

        handler_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                print("[http] " + fmt % args)

            def do_GET(self):
                if self.path == "/health":
                    body = json.dumps({
                        "status": "ok",
                        "loaded": handler_ref.service._loaded,
                    }).encode()
                    self._respond(200, body)
                else:
                    self.send_response(404); self.end_headers()

            def do_POST(self):
                if self.path != "/report":
                    self.send_response(404); self.end_headers(); return

                ctype, pdict = cgi.parse_header(
                    self.headers.get("Content-Type", ""))
                if "multipart/form-data" not in ctype:
                    self.send_response(400); self.end_headers(); return

                pdict["boundary"] = pdict["boundary"].encode()
                pdict["CONTENT-LENGTH"] = int(
                    self.headers.get("Content-Length", 0))
                fields      = cgi.parse_multipart(self.rfile, pdict)
                image_bytes = fields.get("image", [b""])[0]
                form        = {
                    k: (v[0].decode() if isinstance(v[0], bytes) else v[0])
                    for k, v in fields.items() if k != "image"
                }
                status, res = handler_ref._handle_report(image_bytes, form)
                self._respond(
                    status,
                    json.dumps(res, ensure_ascii=False).encode("utf-8"),
                )

            def _respond(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        print(f"[api] Stdlib HTTP server: http://{self.host}:{self.port}")
        HTTPServer((self.host, self.port), _Handler).serve_forever()

    def __repr__(self) -> str:
        return (
            f"MedGemmaAPI("
            f"host={self.host}, port={self.port})"
        )