# -*- coding: utf-8 -*-
"""
src/api/schemas/classification.py
===================================
Pydantic request / response schemas for the /classify endpoint.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    """Per-class softmax probability."""
    class_name:  str
    class_id:    int
    probability: float = Field(..., ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    """Full response returned by POST /classify."""
    request_id:        str
    model:             str = "ConvNeXt-Base"
    image_width:       int
    image_height:      int
    predicted_class:   str   = Field(..., description="Top predicted class name")
    predicted_class_id: int
    confidence:        float = Field(..., ge=0.0, le=1.0,
                                    description="Softmax probability of top class")
    probabilities:     List[ClassProbability] = Field(
        ..., description="Full softmax distribution over all classes"
    )
    class_info:        Dict[str, str] = Field(
        ..., description="Clinical metadata for the predicted class"
    )
    inference_time_ms: float
    warning:           str = (
        "AI output is for research assistance only. "
        "Not a substitute for clinical diagnosis."
    )


class BatchClassificationResponse(BaseModel):
    """Response for POST /classify/batch."""
    total_images: int
    results:      List[ClassificationResponse]
