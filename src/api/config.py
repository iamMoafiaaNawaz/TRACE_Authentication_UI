# -*- coding: utf-8 -*-
"""
src/api/config.py
==================
FastAPI configuration — model path, inference defaults, class names.

All values can be overridden via environment variables so that the same
image works in dev (local best.pt) and production (volume-mounted path).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]  # project root

MODEL_PATH: Path = Path(
    os.environ.get("YOLO_MODEL_PATH",
                   str(_ROOT / "weights" / "yolo" / "yolov11x_best.pt"))
)

CONVNEXT_MODEL_PATH: Path = Path(
    os.environ.get("CONVNEXT_MODEL_PATH",
                   str(_ROOT / "weights" / "convnext" / "best_convnext_checkpoint.pth"))
)

CONVNEXT_IMGSZ: int = int(os.environ.get("CONVNEXT_IMGSZ", "224"))

# ---------------------------------------------------------------------------
# Inference defaults (match training config)
# ---------------------------------------------------------------------------
IMGSZ:        int   = int(os.environ.get("YOLO_IMGSZ",   "640"))
CONF_THRESH:  float = float(os.environ.get("YOLO_CONF",  "0.25"))
IOU_THRESH:   float = float(os.environ.get("YOLO_IOU",   "0.7"))
MAX_DET:      int   = int(os.environ.get("YOLO_MAX_DET", "300"))
DEVICE:       str   = os.environ.get("YOLO_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Class metadata (must match dataset.yaml order)
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = ["BCC", "BKL", "MEL", "NV"]

CLASS_INFO: dict[str, dict] = {
    "BCC": {
        "full_name":  "Basal Cell Carcinoma",
        "risk":       "High",
        "icd10":      "C44.91",
        "action":     "Urgent dermatology referral required",
    },
    "BKL": {
        "full_name":  "Benign Keratosis-like Lesion",
        "risk":       "Low",
        "icd10":      "L82.1",
        "action":     "Monitor; reassess if growth or change observed",
    },
    "MEL": {
        "full_name":  "Melanoma",
        "risk":       "Critical",
        "icd10":      "C43.9",
        "action":     "URGENT: Immediate oncology referral",
    },
    "NV": {
        "full_name":  "Melanocytic Nevus (Benign Mole)",
        "risk":       "Low",
        "icd10":      "D22.9",
        "action":     "Routine monitoring; annual skin check recommended",
    },
}

# ---------------------------------------------------------------------------
# MedGemma
# ---------------------------------------------------------------------------
MEDGEMMA_MODEL_ID: str = os.environ.get(
    "MEDGEMMA_MODEL_ID", "google/medgemma-4b-it"
)
MEDGEMMA_CACHE_DIR: str = os.environ.get(
    "MEDGEMMA_CACHE_DIR", ""       # empty → HuggingFace default ~/.cache/huggingface
)
MEDGEMMA_LOCAL_FILES_ONLY: bool = (
    os.environ.get("MEDGEMMA_LOCAL_FILES_ONLY", "false").lower() == "true"
)
MEDGEMMA_USE_4BIT: bool = (
    os.environ.get("MEDGEMMA_USE_4BIT", "true").lower() == "true"
)

# ---------------------------------------------------------------------------
# API metadata
# ---------------------------------------------------------------------------
API_TITLE:       str = "TRACE Skin Lesion Detection API"
API_VERSION:     str = "1.1.0"
API_DESCRIPTION: str = (
    "TRACE dermoscopy analysis API — four endpoints:\n\n"
    "**POST /detect** — YOLOv11x bounding-box localisation\n\n"
    "**POST /classify** — ConvNeXt-Base 4-class classification\n\n"
    "**POST /xai** — GradCAM++ saliency map + XAI analysis\n\n"
    "**POST /report** — MedGemma humanized clinical narrative\n\n"
    "**Classes:** BCC · BKL · MEL · NV\n\n"
    "> ⚠️ For research use only. Not a substitute for clinical diagnosis."
)
