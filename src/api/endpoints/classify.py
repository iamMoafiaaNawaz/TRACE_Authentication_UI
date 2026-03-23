# -*- coding: utf-8 -*-
"""
src/api/endpoints/classify.py
==============================
POST /classify        — single image 4-class skin lesion classification
POST /classify/batch  — batch classification (up to 32 images)

Model: ConvNeXt-Base trained on ISIC (BCC · BKL · MEL · NV)
Checkpoint: weights/convnext/best_convnext_checkpoint.pth  (.pth format)
"""
from __future__ import annotations

import io
import time
import uuid
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.api import config as cfg
from src.api.dependencies import get_classifier
from src.api.schemas.classification import (
    BatchClassificationResponse,
    ClassificationResponse,
    ClassProbability,
)

router = APIRouter(tags=["Classification"])

# ---------------------------------------------------------------------------
# Inference transforms (must match training: ResizePad 224 + ImageNet norm)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transform(image_size: int = 224) -> transforms.Compose:
    """Aspect-ratio-preserving resize + pad + ImageNet normalisation."""
    from src.preprocessing.transforms import ResizePad
    return transforms.Compose([
        ResizePad(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


_TRANSFORM: Optional[transforms.Compose] = None


def _get_transform() -> transforms.Compose:
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = _build_transform(cfg.CONVNEXT_IMGSZ)
    return _TRANSFORM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")


def _run_classification(model, img: Image.Image) -> tuple[list[float], float]:
    """
    Run ConvNeXt inference on a PIL image.
    Returns (softmax_probs_list, inference_ms).
    """
    tfm    = _get_transform()
    tensor = tfm(img).unsqueeze(0)  # (1, 3, H, W)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)          # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return probs, elapsed_ms


def _build_response(
    img: Image.Image,
    probs: list[float],
    class_names: list[str],
    elapsed_ms: float,
) -> ClassificationResponse:
    W, H = img.size

    top_idx  = int(max(range(len(probs)), key=lambda i: probs[i]))
    top_name = class_names[top_idx]
    top_conf = round(probs[top_idx], 4)

    prob_list = [
        ClassProbability(
            class_name  = class_names[i],
            class_id    = i,
            probability = round(p, 4),
        )
        for i, p in enumerate(probs)
    ]

    class_info = cfg.CLASS_INFO.get(top_name, {
        "full_name": top_name,
        "risk":      "Unknown",
        "icd10":     "N/A",
        "action":    "Consult dermatologist",
    })

    return ClassificationResponse(
        request_id         = str(uuid.uuid4()),
        image_width        = W,
        image_height       = H,
        predicted_class    = top_name,
        predicted_class_id = top_idx,
        confidence         = top_conf,
        probabilities      = prob_list,
        class_info         = class_info,
        inference_time_ms  = round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a single dermoscopy image (ConvNeXt-Base)",
    description=(
        "Upload a dermoscopy image and receive a 4-class classification.\n\n"
        "**Classes:** BCC (Basal Cell Carcinoma) · BKL (Benign Keratosis) · "
        "MEL (Melanoma) · NV (Melanocytic Nevus)\n\n"
        "- Accepts JPEG / PNG / BMP / TIFF\n"
        "- Returns softmax probabilities for all classes\n"
        "- Checkpoint: `weights/convnext/best_convnext_checkpoint.pth`"
    ),
)
async def classify_single(
    image: UploadFile = File(..., description="Dermoscopy image file"),
    classifier = Depends(get_classifier),
) -> ClassificationResponse:
    model, class_names = classifier
    raw               = await image.read()
    img               = _parse_image(raw)
    probs, elapsed_ms = _run_classification(model, img)
    return _build_response(img, probs, class_names, elapsed_ms)


@router.post(
    "/classify/batch",
    response_model=BatchClassificationResponse,
    summary="Classify multiple dermoscopy images (ConvNeXt-Base)",
    description="Upload up to 32 images; each is classified independently.",
)
async def classify_batch(
    images: List[UploadFile] = File(..., description="One or more dermoscopy images"),
    classifier = Depends(get_classifier),
) -> BatchClassificationResponse:
    if len(images) > 32:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 32 images per request.",
        )

    model, class_names = classifier
    all_results: List[ClassificationResponse] = []

    for img_file in images:
        raw               = await img_file.read()
        img               = _parse_image(raw)
        probs, elapsed_ms = _run_classification(model, img)
        all_results.append(_build_response(img, probs, class_names, elapsed_ms))

    return BatchClassificationResponse(
        total_images = len(all_results),
        results      = all_results,
    )
