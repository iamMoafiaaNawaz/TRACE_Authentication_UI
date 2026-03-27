import os
from glob import glob
from typing import Any, Dict, Optional

import numpy as np

# Ultralytics tries to create settings under %APPDATA% by default; in some locked-down
# environments that's not writable. Force a local, writable config root.
_LOCAL_YOLO_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", ".yolo_config")
_LOCAL_YOLO_CONFIG_DIR = os.path.abspath(_LOCAL_YOLO_CONFIG_DIR)
os.makedirs(_LOCAL_YOLO_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _LOCAL_YOLO_CONFIG_DIR)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
YOLO_MODEL_PATH = None

_yolo_model = None
_yolo_error: Optional[str] = None

def _resolve_yolo_model_path() -> str:
    """Resolve preferred YOLO model path from model directory.

    Priority:
      1) `YOLO_MODEL_PATH` env var if valid.
      2) Custom lesion checkpoints in model dir: best.pt / best.pt.zip / best*.pt / best*.pth.
      3) fallback to `model/yolo_lesion_model.pt`.
    """
    env_path = os.getenv("YOLO_MODEL_PATH")
    if env_path:
        env_abs = os.path.abspath(env_path)
        if os.path.exists(env_abs):
            return env_abs

    custom_candidates = []
    # Prefer explicit best.pt first for custom lesion weights.
    for name in ["best.pt", "best.pt.zip"]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            custom_candidates.append(os.path.abspath(p))
    for p in sorted(glob(os.path.join(MODEL_DIR, "best*.pt"))):
        ap = os.path.abspath(p)
        if ap not in custom_candidates:
            custom_candidates.append(ap)
    for p in sorted(glob(os.path.join(MODEL_DIR, "best*.pth"))):
        ap = os.path.abspath(p)
        if ap not in custom_candidates:
            custom_candidates.append(ap)
    if custom_candidates:
        return custom_candidates[0]

    fallback = os.path.join(MODEL_DIR, "yolo_lesion_model.pt")
    return os.path.abspath(fallback)


def load_yolo_once() -> None:
    global _yolo_model, _yolo_error, YOLO_MODEL_PATH
    if _yolo_model is not None:
        return
    try:
        from ultralytics import YOLO  # type: ignore

        primary = _resolve_yolo_model_path()
        fallback = os.path.abspath(os.path.join(MODEL_DIR, "yolo_lesion_model.pt"))
        candidates = [primary]
        if fallback not in candidates and os.path.exists(fallback):
            candidates.append(fallback)

        last_exc = None
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                _yolo_model = YOLO(path, task="detect")
                YOLO_MODEL_PATH = path
                _yolo_error = None
                print(f"YOLO model loaded from: {path}")
                return
            except Exception as exc:
                last_exc = exc
                print(f"YOLO load failed for {path}: {exc}")

        _yolo_model = None
        _yolo_error = str(last_exc) if last_exc else "No valid YOLO model file found"
    except Exception as exc:
        _yolo_model = None
        _yolo_error = str(exc)


def get_yolo_status() -> Dict[str, Any]:
    resolved = YOLO_MODEL_PATH or _resolve_yolo_model_path()
    return {"loaded": _yolo_model is not None, "error": _yolo_error, "model_path": resolved}


def localize_lesion(image_bgr, conf: float = 0.15) -> Dict[str, Any]:
    """Run YOLO localization and return raw detections without custom filtering.

    Returns a dict safe for API responses:
      - status: "success" | "no_detection" | "failed"
      - box_found: bool
      - detections: list of raw model detections (bbox/conf/class/label)
      - bbox_normalized/confidence/class_id/label: first detection fields for backward compatibility
      - error: optional str
    """
    if _yolo_model is None:
        return {
            "status": "failed",
            "box_found": False,
            "detections": [],
            "bbox_normalized": None,
            "confidence": None,
            "class_id": None,
            "label": None,
            "error": _yolo_error or "YOLO not loaded",
        }

    try:
        if image_bgr is None:
            return {
                "status": "failed",
                "box_found": False,
                "detections": [],
                "bbox_normalized": None,
                "confidence": None,
                "class_id": None,
                "label": None,
                "error": "YOLO input image is None",
            }

        # Enforce color image for detector quality (convert grayscale -> BGR).
        if len(image_bgr.shape) == 2:
            image_bgr = np.stack([image_bgr, image_bgr, image_bgr], axis=-1)
        elif len(image_bgr.shape) == 3 and image_bgr.shape[2] == 1:
            image_bgr = np.concatenate([image_bgr, image_bgr, image_bgr], axis=2)

        # Use a practical confidence threshold to surface lesion detections.
        results = _yolo_model.predict(image_bgr, conf=float(conf), verbose=False)
        if not results or len(results) < 1 or getattr(results[0], "boxes", None) is None:
            return {
                "status": "no_detection",
                "box_found": False,
                "detections": [],
                "bbox_normalized": None,
                "confidence": None,
                "class_id": None,
                "label": None,
            }

        # Required raw diagnostics BEFORE any formatting/filtering.
        try:
            print("CRITICAL DEBUG: Total raw detections found by YOLO model BEFORE any custom filters: ", len(results[0].boxes))
        except Exception as dbg_exc:
            print(f"CRITICAL DEBUG: failed to print raw detections: {dbg_exc}")

        res0 = results[0]

        boxes = res0.boxes
        if boxes is None or len(boxes) == 0:
            return {
                "status": "no_detection",
                "box_found": False,
                "detections": [],
                "bbox_normalized": None,
                "confidence": None,
                "class_id": None,
                "label": None,
            }

        xyxy = boxes.xyxy.detach().float().cpu().numpy()
        confs = boxes.conf.detach().float().cpu().numpy()
        cls_ids = boxes.cls.detach().float().cpu().numpy() if getattr(boxes, "cls", None) is not None else np.array([])
        if getattr(boxes, "xyxyn", None) is not None:
            xyxyn = boxes.xyxyn.detach().float().cpu().numpy()
        else:
            h, w = image_bgr.shape[:2]
            if h > 0 and w > 0:
                xyxyn = xyxy.copy()
                xyxyn[:, [0, 2]] /= float(w)
                xyxyn[:, [1, 3]] /= float(h)
            else:
                xyxyn = xyxy.copy()

        names = getattr(res0, "names", None)
        detections = []
        for i in range(len(boxes)):
            cid = int(cls_ids[i]) if i < len(cls_ids) else None
            label = None
            if isinstance(names, dict) and cid is not None:
                label = names.get(cid)
            elif isinstance(names, list) and cid is not None and 0 <= cid < len(names):
                label = names[cid]

            det = {
                "bbox": [float(v) for v in xyxy[i].tolist()[:4]],
                "bbox_normalized": [float(v) for v in xyxyn[i].tolist()[:4]],
                "confidence": float(confs[i]) if i < len(confs) else None,
                "class_id": cid,
                "label": str(label) if label is not None else None,
            }
            detections.append(det)

        first = detections[0] if detections else None
        # Backward compatibility + explicit coordinates for frontend consumers.
        x_min = y_min = x_max = y_max = None
        if first and isinstance(first.get("bbox"), list) and len(first["bbox"]) == 4:
            x_min, y_min, x_max, y_max = first["bbox"]
        return {
            "status": "success" if detections else "no_detection",
            "box_found": bool(detections),
            "detections": detections,
            "bbox_normalized": first.get("bbox_normalized") if first else None,
            "bbox": first.get("bbox") if first else None,
            "xmin": x_min,
            "ymin": y_min,
            "xmax": x_max,
            "ymax": y_max,
            "confidence": first.get("confidence") if first else None,
            "class_id": first.get("class_id") if first else None,
            "label": first.get("label") if first else None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "box_found": False,
            "detections": [],
            "bbox_normalized": None,
            "confidence": None,
            "class_id": None,
            "label": None,
            "error": str(exc),
        }
