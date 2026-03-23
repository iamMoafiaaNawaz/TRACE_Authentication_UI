# -*- coding: utf-8 -*-
"""
src/api/dependencies.py
========================
FastAPI dependency — singleton YOLO model loader.

The model is loaded once on application startup and injected into every
request handler via FastAPI's dependency injection system.

Usage in endpoint
-----------------
>>> @router.post("/detect")
... async def detect(model: YOLO = Depends(get_model), ...):
...     results = model.predict(...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.api import config as cfg


class _ModelStore:
    """Process-level singleton holding the loaded YOLO model."""
    _model = None
    _loaded: bool = False

    @classmethod
    def load(cls) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        if not Path(cfg.MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model weights not found: {cfg.MODEL_PATH}\n"
                f"Place best.pt at: {cfg.MODEL_PATH}"
            )
        cls._model  = YOLO(str(cfg.MODEL_PATH))
        cls._loaded = True

    @classmethod
    def get(cls):
        if not cls._loaded:
            cls.load()
        return cls._model

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded


def get_model():
    """FastAPI dependency — returns the loaded YOLO model."""
    return _ModelStore.get()


# ---------------------------------------------------------------------------
# ConvNeXt classifier store
# ---------------------------------------------------------------------------

class _ClassifierStore:
    """Process-level singleton holding the loaded ConvNeXt model + metadata."""
    _model:       Optional[object] = None
    _class_names: Optional[list]   = None
    _loaded:      bool             = False

    @classmethod
    def load(cls) -> None:
        import torch
        from src.models.classifier import ConvNeXtClassifier

        if not Path(cfg.CONVNEXT_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"ConvNeXt weights not found: {cfg.CONVNEXT_MODEL_PATH}\n"
                f"Place checkpoint at: {cfg.CONVNEXT_MODEL_PATH}"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, class_names, _ = ConvNeXtClassifier.load_checkpoint(
            cfg.CONVNEXT_MODEL_PATH, device
        )
        cls._model       = model
        cls._class_names = class_names or cfg.CLASS_NAMES
        cls._loaded      = True

    @classmethod
    def get(cls):
        if not cls._loaded:
            cls.load()
        return cls._model, cls._class_names

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded


def get_classifier():
    """FastAPI dependency — returns (ConvNeXt model, class_names)."""
    return _ClassifierStore.get()


# ---------------------------------------------------------------------------
# MedGemma service store
# ---------------------------------------------------------------------------

class _MedGemmaStore:
    """Process-level singleton for the MedGemmaService (lazy-loaded on first /report call)."""
    _service = None

    @classmethod
    def load(cls) -> None:
        from src.models.medgemma import MedGemmaService
        cls._service = MedGemmaService(
            model_id         = cfg.MEDGEMMA_MODEL_ID,
            cache_dir        = cfg.MEDGEMMA_CACHE_DIR or None,
            use_4bit         = cfg.MEDGEMMA_USE_4BIT,
            local_files_only = cfg.MEDGEMMA_LOCAL_FILES_ONLY,
        )

    @classmethod
    def get(cls):
        if cls._service is None:
            cls.load()
        return cls._service

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._service is not None and getattr(cls._service, "_loaded", False)


def get_medgemma():
    """FastAPI dependency — returns the MedGemmaService.

    Raises RuntimeError wrapping any ImportError / OSError so the
    /report endpoint can return a readable 503.
    """
    try:
        return _MedGemmaStore.get()
    except (ImportError, OSError) as e:
        raise RuntimeError(str(e))
