# -*- coding: utf-8 -*-
"""
src/api/app.py
==============
FastAPI application factory for the TRACE Skin Lesion Detection API.

Creates and configures the FastAPI app with all routers, startup events,
and CORS middleware.

Usage
-----
    # Run with uvicorn (from project root):
    uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

    # Or via the project entry point:
    python main_api.py

MedGemma re-exports (for backwards compatibility)
--------------------------------------------------
    from src.api.app import MedGemmaService, MedGemmaAPI
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import config as cfg
from src.api.dependencies import _ClassifierStore, _ModelStore
from src.api.endpoints import classify, detect, health, report, xai

# Backwards-compat re-exports (MedGemma API server)
from src.models.medgemma import MedGemmaAPI, MedGemmaService  # noqa: F401

__all__ = ["create_app", "MedGemmaService", "MedGemmaAPI"]


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Returns a fully-configured FastAPI instance.  Designed to be called
    by uvicorn ``--factory`` flag or by ``main_api.py``.
    """
    app = FastAPI(
        title       = cfg.API_TITLE,
        version     = cfg.API_VERSION,
        description = cfg.API_DESCRIPTION,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        openapi_url = "/openapi.json",
    )

    # --- CORS (allow all origins for research/dev; restrict in production) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # --- Startup: load both models once ---
    @app.on_event("startup")
    async def _load_models() -> None:
        try:
            _ModelStore.load()
            print(f"[startup] YOLO loaded: {cfg.MODEL_PATH}")
        except FileNotFoundError as e:
            print(f"[startup] WARNING (YOLO): {e}")
            print("[startup] /detect will fail until YOLO weights are placed")

        try:
            _ClassifierStore.load()
            print(f"[startup] ConvNeXt loaded: {cfg.CONVNEXT_MODEL_PATH}")
        except FileNotFoundError as e:
            print(f"[startup] WARNING (ConvNeXt): {e}")
            print("[startup] /classify will fail until ConvNeXt weights are placed")

    # --- Routers ---
    app.include_router(health.router)
    app.include_router(detect.router)
    app.include_router(classify.router)
    app.include_router(xai.router)
    app.include_router(report.router)

    return app


# Module-level app instance (for `uvicorn src.api.app:app`)
app = create_app()
