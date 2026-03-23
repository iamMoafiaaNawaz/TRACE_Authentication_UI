# -*- coding: utf-8 -*-
"""
src/api/endpoints/xai.py
=========================
POST /xai  — GradCAM++ saliency map + XAI analysis for a single image.

Pipeline
--------
1. Decode uploaded image
2. Apply ConvNeXt eval transforms (ResizePad 224 + ImageNet normalise)
3. Forward + backward pass with hooks → GradCAM++ saliency map
4. Second forward pass (no_grad) → softmax probabilities
5. Compose side-by-side overlay: original | jet-coloured blend
6. Return overlay as base64 PNG + saliency stats + classification

No extra weights required — uses the same ConvNeXt checkpoint as /classify.
"""
from __future__ import annotations

import base64
import io
import time
import uuid
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm_module
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.api import config as cfg
from src.api.dependencies import get_classifier
from src.api.schemas.xai import ClassProbabilityXAI, GradCAMAnalysis, XAIResponse
from src.xai.gradcam import GradCAMPlusPlus, get_gradcam_target_layer

router = APIRouter(tags=["XAI"])

# ---------------------------------------------------------------------------
# Eval transform (matches training pipeline)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_MEAN_T = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_STD_T  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
_TRANSFORM: Optional[transforms.Compose] = None


def _get_transform() -> transforms.Compose:
    global _TRANSFORM
    if _TRANSFORM is None:
        from src.preprocessing.transforms import ResizePad
        _TRANSFORM = transforms.Compose([
            ResizePad(cfg.CONVNEXT_IMGSZ),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return _TRANSFORM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Un-normalise a (3, H, W) CPU tensor to float [0,1] RGB numpy."""
    return (tensor.cpu() * _STD_T + _MEAN_T).clamp(0, 1).permute(1, 2, 0).numpy()


def _make_overlay_b64(rgb: np.ndarray, cam: np.ndarray,
                      pred_class: str, confidence: float) -> str:
    """Side-by-side original | jet overlay → base64 PNG."""
    cam_rgb = cm_module.get_cmap("jet")(cam)[:, :, :3]
    overlay = (0.55 * rgb + 0.45 * cam_rgb).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb);    axes[0].set_title("Original", fontsize=10);   axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title(
        f"GradCAM++  →  {pred_class}  ({confidence:.1%})", fontsize=10
    );  axes[1].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _analyse_cam(cam: np.ndarray, pred_class: str) -> dict:
    """Quantitative saliency statistics (mirrors GradCAMPlusPlus.analyse)."""
    h, w  = cam.shape
    high  = (cam >= 0.70).sum() / cam.size * 100
    mid   = ((cam >= 0.40) & (cam < 0.70)).sum() / cam.size * 100
    ys, xs = np.where(cam >= 0.70)

    if len(ys):
        cy = float(ys.mean()) / h
        cx = float(xs.mean()) / w
        v  = "upper"   if cy < 0.4 else ("lower"  if cy > 0.6 else "central")
        hz = "left"    if cx < 0.4 else ("right"  if cx > 0.6 else "central")
        region = f"{v}-{hz}" if not (v == "central" and hz == "central") else "central"
    else:
        region = "diffuse"

    peak     = float(cam.max())
    strength = "Strong" if peak > 0.8 else ("Moderate" if peak > 0.5 else "Weak")
    return {
        "high_activation_pct": round(high, 2),
        "mid_activation_pct":  round(mid, 2),
        "mean_activation":     round(float(cam.mean()), 4),
        "peak_activation":     round(peak, 4),
        "primary_region":      region,
        "xai_summary": (
            f"GradCAM++ for '{pred_class}': focus on {region} region. "
            f"{high:.1f}% pixels >70% activation. Peak={peak:.3f}. "
            f"{strength} spatial evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/xai",
    response_model=XAIResponse,
    summary="GradCAM++ explainability for a single dermoscopy image",
    description=(
        "Upload a dermoscopy image and receive:\n\n"
        "- **Classification** — same as `/classify`\n"
        "- **GradCAM++ overlay** — original side-by-side with saliency heatmap (base64 PNG)\n"
        "- **Saliency statistics** — activation %, primary region, peak value\n\n"
        "**No extra weights required.** Uses the same ConvNeXt checkpoint as `/classify`.\n\n"
        "Decode the `overlay_base64` field in Python:\n"
        "```python\n"
        "import base64, io\nfrom PIL import Image\n"
        "img = Image.open(io.BytesIO(base64.b64decode(result['overlay_base64'])))\n"
        "img.show()\n"
        "```"
    ),
)
async def xai_single(
    image:      UploadFile = File(..., description="Dermoscopy image (JPEG/PNG/BMP/TIFF)"),
    classifier  = Depends(get_classifier),
) -> XAIResponse:
    model, class_names = classifier
    raw  = await image.read()
    pil  = _parse_image(raw)
    W, H = pil.size

    tfm    = _get_transform()
    tensor = tfm(pil).unsqueeze(0).to(next(model.parameters()).device)  # (1,3,H,W)

    t0 = time.perf_counter()
    model.eval()

    # Step 1 — GradCAM++ (needs gradient flow)
    cam_gen = GradCAMPlusPlus(model, get_gradcam_target_layer(model))
    try:
        cam = cam_gen(tensor)           # (Hs, Ws) float [0,1], does forward+backward
    finally:
        cam_gen.remove()                # always remove hooks

    # Step 2 — full probabilities (no grad needed)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().tolist()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    top_idx  = int(max(range(len(probs)), key=lambda i: probs[i]))
    top_name = class_names[top_idx] if top_idx < len(class_names) else f"cls_{top_idx}"
    top_conf = round(probs[top_idx], 4)

    analysis   = _analyse_cam(cam, top_name)
    rgb        = _tensor_to_rgb(tensor.squeeze(0))
    ovl_b64    = _make_overlay_b64(rgb, cam, top_name, top_conf)
    class_info = cfg.CLASS_INFO.get(top_name, {
        "full_name": top_name, "risk": "Unknown",
        "icd10": "N/A",        "action": "Consult dermatologist",
    })

    return XAIResponse(
        request_id         = str(uuid.uuid4()),
        image_width        = W,
        image_height       = H,
        predicted_class    = top_name,
        predicted_class_id = top_idx,
        confidence         = top_conf,
        probabilities      = [
            ClassProbabilityXAI(class_name=class_names[i], class_id=i, probability=round(p, 4))
            for i, p in enumerate(probs)
        ],
        class_info         = class_info,
        gradcam_analysis   = GradCAMAnalysis(**analysis),
        overlay_base64     = ovl_b64,
        inference_time_ms  = round(elapsed_ms, 2),
    )
