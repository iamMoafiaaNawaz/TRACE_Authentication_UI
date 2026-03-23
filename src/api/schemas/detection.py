# -*- coding: utf-8 -*-
"""
src/api/schemas/detection.py
=============================
Pydantic request / response schemas for the /detect endpoint.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel-space bounding box (x1y1x2y2) + normalised centre+size."""
    x1: float = Field(..., description="Left edge in pixels")
    y1: float = Field(..., description="Top edge in pixels")
    x2: float = Field(..., description="Right edge in pixels")
    y2: float = Field(..., description="Bottom edge in pixels")
    cx_norm: float = Field(..., description="Normalised centre-x  (0-1)")
    cy_norm: float = Field(..., description="Normalised centre-y  (0-1)")
    w_norm:  float = Field(..., description="Normalised width     (0-1)")
    h_norm:  float = Field(..., description="Normalised height    (0-1)")


class ClassInfo(BaseModel):
    full_name:  str
    risk:       str
    icd10:      str
    action:     str


class Detection(BaseModel):
    """Single detected lesion."""
    detection_id:  int
    class_id:      int   = Field(..., description="0=BCC 1=BKL 2=MEL 3=NV")
    class_name:    str
    confidence:    float = Field(..., ge=0.0, le=1.0)
    box:           BoundingBox
    class_info:    ClassInfo


class DetectionResponse(BaseModel):
    """Full response returned by POST /detect."""
    request_id:        str
    model:             str = "YOLOv11x"
    image_width:       int
    image_height:      int
    num_detections:    int
    detections:        List[Detection]
    conf_threshold:    float
    iou_threshold:     float
    inference_time_ms: float
    warning:           str = (
        "AI output is for research assistance only. "
        "Not a substitute for clinical diagnosis."
    )


class HealthResponse(BaseModel):
    status:                  str
    yolo_loaded:             bool
    yolo_path:               str
    convnext_loaded:         bool
    convnext_path:           str
    medgemma_loaded:         bool
    medgemma_model_id:       str
    classes:                 List[str]
    device:                  str
    version:                 str


class BatchDetectionResponse(BaseModel):
    """Response for POST /detect/batch."""
    total_images:   int
    results:        List[DetectionResponse]
