# -*- coding: utf-8 -*-
"""
src/api/endpoints/report.py
============================
POST /report  — AI-generated humanized clinical report via MedGemma-4B.

The report is written in plain, flowing paragraphs readable by both
clinicians and patients — no numbered lists or bullet points.

Prerequisites
-------------
MedGemma is a gated model on HuggingFace. Before the first /report call:

  1. Accept terms at: https://huggingface.co/google/medgemma-4b-it
  2. pip install transformers accelerate bitsandbytes
  3. huggingface-cli login   (paste your HF access token)

The model (~8 GB in 4-bit) is downloaded automatically on the first request
and cached to MEDGEMMA_CACHE_DIR (default: ~/.cache/huggingface).

Set the env var MEDGEMMA_LOCAL_FILES_ONLY=true to skip downloads once cached.
"""
from __future__ import annotations

import io
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api import config as cfg
from src.api.dependencies import get_medgemma
from src.api.schemas.report import ReportResponse

router = APIRouter(tags=["Clinical Report"])


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/report",
    response_model=ReportResponse,
    summary="Generate a humanized clinical report via MedGemma",
    description=(
        "Upload a dermoscopy image together with the detection/classification result "
        "to receive a **plain-English clinical narrative** readable by both doctors "
        "and patients.\n\n"
        "The report is written in flowing paragraphs covering:\n"
        "- What was found and where\n"
        "- Visible features and dermoscopic criteria\n"
        "- Risk level and justification\n"
        "- Recommended next clinical step\n"
        "- AI limitation reminder\n\n"
        "**Prerequisites:** MedGemma-4B must be downloaded first. "
        "See `docs/yolo_api.md` → *MedGemma Setup* for instructions.\n\n"
        "**Typical generation time:** 15–45 s on CPU, 3–8 s on GPU."
    ),
)
async def generate_report(
    image:      UploadFile = File(...,  description="Dermoscopy image (JPEG/PNG/BMP/TIFF)"),
    pred_class: str        = Form(...,  description="Predicted class: BCC | BKL | MEL | NV"),
    pred_conf:  float      = Form(0.5,  description="Model confidence score (0–1)"),
    box_cx:     float      = Form(0.5,  description="Normalised bounding box centre-x (0–1)"),
    box_cy:     float      = Form(0.5,  description="Normalised bounding box centre-y (0–1)"),
    box_w:      float      = Form(0.4,  description="Normalised bounding box width  (0–1)"),
    box_h:      float      = Form(0.4,  description="Normalised bounding box height (0–1)"),
) -> ReportResponse:

    # validate class
    if pred_class not in cfg.CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"pred_class must be one of {cfg.CLASS_NAMES}, got '{pred_class}'",
        )

    # decode image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")

    # get MedGemma service (lazy-loaded)
    try:
        service = get_medgemma()
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"MedGemma is not available: {e}\n\n"
                "Setup instructions:\n"
                "1. Accept model terms at https://huggingface.co/google/medgemma-4b-it\n"
                "2. pip install transformers accelerate bitsandbytes\n"
                "3. huggingface-cli login\n"
                "4. Restart the API server"
            ),
        )

    # generate report
    try:
        result = service.generate_report(
            image_path = raw,
            pred_class = pred_class,
            pred_conf  = pred_conf,
            box_cx     = box_cx,
            box_cy     = box_cy,
            box_w      = box_w,
            box_h      = box_h,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return ReportResponse(
        request_id   = str(uuid.uuid4()),
        pred_class   = result["pred_class"],
        pred_conf    = result["pred_conf"],
        box_pixels   = result["box_pixels"],
        image_size   = result["image_size"],
        report       = result["report"],
        gen_time_sec = result["gen_time_sec"],
    )
