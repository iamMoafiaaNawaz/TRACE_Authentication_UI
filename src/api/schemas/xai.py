# -*- coding: utf-8 -*-
"""
src/api/schemas/xai.py
=======================
Pydantic response schemas for the /xai endpoint.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class GradCAMAnalysis(BaseModel):
    """Statistics extracted from the GradCAM++ saliency map."""
    high_activation_pct: float = Field(..., description="% pixels with activation >= 0.70")
    mid_activation_pct:  float = Field(..., description="% pixels with activation 0.40-0.70")
    mean_activation:     float
    peak_activation:     float
    primary_region:      str   = Field(..., description="Spatial region of peak attention (e.g. central, upper-left)")
    xai_summary:         str   = Field(..., description="One-line human-readable summary")


class ClassProbabilityXAI(BaseModel):
    class_name:  str
    class_id:    int
    probability: float


class XAIResponse(BaseModel):
    """Full response returned by POST /xai."""
    request_id:         str
    model:              str = "ConvNeXt-Base + GradCAM++"
    image_width:        int
    image_height:       int
    predicted_class:    str
    predicted_class_id: int
    confidence:         float
    probabilities:      List[ClassProbabilityXAI]
    class_info:         Dict[str, str]
    gradcam_analysis:   GradCAMAnalysis
    overlay_base64:     str  = Field(..., description="Base64-encoded PNG: original | GradCAM++ overlay")
    inference_time_ms:  float
    warning:            str  = (
        "AI output is for research assistance only. "
        "Not a substitute for clinical diagnosis."
    )
