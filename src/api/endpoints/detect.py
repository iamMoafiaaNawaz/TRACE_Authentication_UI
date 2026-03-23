# -*- coding: utf-8 -*-
"""
src/api/endpoints/detect.py
=============================
POST /detect        — single image lesion detection
POST /detect/batch  — batch image detection
"""
from __future__ import annotations

import io
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api import config as cfg
from src.api.dependencies import get_model
from src.api.schemas.detection import (
    BatchDetectionResponse,
    BoundingBox,
    ClassInfo,
    Detection,
    DetectionResponse,
)

router = APIRouter(tags=["Detection"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_image(raw: bytes) -> Image.Image:
    """Decode uploaded bytes to a PIL Image."""
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")


def _run_inference(
    model,
    img: Image.Image,
    conf: float,
    iou: float,
) -> tuple:
    """
    Run YOLO inference.
    Returns (results, inference_time_ms).
    """
    t0 = time.perf_counter()
    results = model.predict(
        source  = img,
        imgsz   = cfg.IMGSZ,
        conf    = conf,
        iou     = iou,
        max_det = cfg.MAX_DET,
        device  = cfg.DEVICE,
        verbose = False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return results, elapsed_ms


def _build_detections(results, W: int, H: int) -> List[Detection]:
    """Convert ultralytics Results -> list of Detection schema objects."""
    detections: List[Detection] = []

    if not results or results[0].boxes is None:
        return detections

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()   # (N, 4)
    confs      = results[0].boxes.conf.cpu().numpy()    # (N,)
    cls_ids    = results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)

    for i, (xyxy, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, cls_ids)):
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

        cx_norm = ((x1 + x2) / 2.0) / W
        cy_norm = ((y1 + y2) / 2.0) / H
        w_norm  = (x2 - x1) / W
        h_norm  = (y2 - y1) / H

        cls_name = cfg.CLASS_NAMES[cls_id] if cls_id < len(cfg.CLASS_NAMES) else f"cls_{cls_id}"
        info     = cfg.CLASS_INFO.get(cls_name, {
            "full_name": cls_name, "risk": "Unknown",
            "icd10": "N/A",       "action": "Consult dermatologist",
        })

        detections.append(Detection(
            detection_id = i,
            class_id     = int(cls_id),
            class_name   = cls_name,
            confidence   = round(float(conf), 4),
            box          = BoundingBox(
                x1=round(x1, 1), y1=round(y1, 1),
                x2=round(x2, 1), y2=round(y2, 1),
                cx_norm=round(cx_norm, 4), cy_norm=round(cy_norm, 4),
                w_norm=round(w_norm, 4),   h_norm=round(h_norm, 4),
            ),
            class_info = ClassInfo(**info),
        ))

    return detections


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect skin lesions in a single image",
    description=(
        "Upload a dermoscopy image and receive bounding box predictions "
        "for BCC, BKL, MEL, and NV lesions.\n\n"
        "- Accepts JPEG / PNG / BMP / TIFF\n"
        "- Returns normalised (0-1) and pixel-space coordinates\n"
        "- `conf` and `iou` override global defaults if provided"
    ),
)
async def detect_single(
    image: UploadFile = File(..., description="Dermoscopy image file"),
    conf:  Optional[float] = Form(None, ge=0.01, le=1.0,
                                  description="Confidence threshold (default 0.25)"),
    iou:   Optional[float] = Form(None, ge=0.1,  le=1.0,
                                  description="IoU NMS threshold (default 0.7)"),
    model = Depends(get_model),
) -> DetectionResponse:
    raw  = await image.read()
    img  = _parse_image(raw)
    W, H = img.size

    conf_used = conf if conf is not None else cfg.CONF_THRESH
    iou_used  = iou  if iou  is not None else cfg.IOU_THRESH

    results, elapsed_ms = _run_inference(model, img, conf_used, iou_used)
    detections          = _build_detections(results, W, H)

    return DetectionResponse(
        request_id        = str(uuid.uuid4()),
        image_width       = W,
        image_height      = H,
        num_detections    = len(detections),
        detections        = detections,
        conf_threshold    = conf_used,
        iou_threshold     = iou_used,
        inference_time_ms = round(elapsed_ms, 2),
    )


@router.post(
    "/detect/batch",
    response_model=BatchDetectionResponse,
    summary="Detect lesions in multiple images",
    description=(
        "Upload up to 32 images and receive detection results for each.\n\n"
        "Images are processed sequentially on the same device."
    ),
)
async def detect_batch(
    images: List[UploadFile] = File(..., description="One or more dermoscopy images"),
    conf:   Optional[float]  = Form(None, ge=0.01, le=1.0),
    iou:    Optional[float]  = Form(None, ge=0.1,  le=1.0),
    model = Depends(get_model),
) -> BatchDetectionResponse:
    if len(images) > 32:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 32 images per request.",
        )

    conf_used = conf if conf is not None else cfg.CONF_THRESH
    iou_used  = iou  if iou  is not None else cfg.IOU_THRESH

    all_results: List[DetectionResponse] = []

    for img_file in images:
        raw  = await img_file.read()
        img  = _parse_image(raw)
        W, H = img.size

        results, elapsed_ms = _run_inference(model, img, conf_used, iou_used)
        detections          = _build_detections(results, W, H)

        all_results.append(DetectionResponse(
            request_id        = str(uuid.uuid4()),
            image_width       = W,
            image_height      = H,
            num_detections    = len(detections),
            detections        = detections,
            conf_threshold    = conf_used,
            iou_threshold     = iou_used,
            inference_time_ms = round(elapsed_ms, 2),
        ))

    return BatchDetectionResponse(
        total_images = len(all_results),
        results      = all_results,
    )
