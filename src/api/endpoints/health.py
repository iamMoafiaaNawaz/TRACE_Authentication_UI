# -*- coding: utf-8 -*-
"""
src/api/endpoints/health.py
=============================
GET /health — liveness + model status for all four models.
"""
from fastapi import APIRouter
from src.api.schemas.detection import HealthResponse
from src.api import config as cfg
from src.api.dependencies import _ClassifierStore, _MedGemmaStore, _ModelStore

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check — all four models",
    description=(
        "Returns liveness and load status for all four pipeline models:\n\n"
        "- **YOLO** (`/detect`) — YOLOv11x bounding-box localisation\n"
        "- **ConvNeXt** (`/classify`, `/xai`) — 4-class skin lesion classifier\n"
        "- **MedGemma** (`/report`) — clinical report generation\n\n"
        "`medgemma_loaded` is `false` until the first `/report` call "
        "(MedGemma is lazy-loaded on demand)."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse(
        status            = "ok",
        yolo_loaded       = _ModelStore.is_loaded(),
        yolo_path         = str(cfg.MODEL_PATH),
        convnext_loaded   = _ClassifierStore.is_loaded(),
        convnext_path     = str(cfg.CONVNEXT_MODEL_PATH),
        medgemma_loaded   = _MedGemmaStore.is_loaded(),
        medgemma_model_id = cfg.MEDGEMMA_MODEL_ID,
        classes           = cfg.CLASS_NAMES,
        device            = cfg.DEVICE,
        version           = cfg.API_VERSION,
    )
