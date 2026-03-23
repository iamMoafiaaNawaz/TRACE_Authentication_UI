# -*- coding: utf-8 -*-
"""
src/api/schemas/report.py
==========================
Pydantic response schemas for the /report (MedGemma) endpoint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ReportResponse(BaseModel):
    """Full response returned by POST /report."""
    request_id:    str
    model:         str = "MedGemma-4B"
    pred_class:    str
    pred_conf:     float
    box_pixels:    dict
    image_size:    dict
    report:        str  = Field(..., description="Humanized clinical narrative in plain paragraphs")
    gen_time_sec:  float
    warning:       str  = (
        "This report is AI-generated for clinical decision support only. "
        "It does not constitute a medical diagnosis. "
        "All findings must be reviewed and confirmed by a qualified dermatologist."
    )


class ReportUnavailableResponse(BaseModel):
    """Returned when MedGemma is not loaded (model not downloaded)."""
    error:        str
    instructions: str
