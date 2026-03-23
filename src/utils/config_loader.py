# This file will load YAML configuration files for experiments
# -*- coding: utf-8 -*-
"""
src/utils/config_loader.py
==========================
YAML / argparse configuration loader for the TRACE pipeline.

Provides a single ``Config`` dataclass-style object that merges
defaults, a YAML file, and CLI overrides — in that priority order.
"""

import argparse
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


# ==============================================================================
# CONFIG DATACLASS
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    All hyper-parameters and paths for one training run.

    Attributes are grouped by concern for readability.
    Defaults mirror the values used in the original ConvNeXt experiment.
    """

    # ---- Paths ---------------------------------------------------------------
    data_root:  str = "/home/f223085/ExperiemntNo2/dataset_split"
    output_dir: str = "./convNext"

    # ---- Data / pre-processing -----------------------------------------------
    image_size:  int = 512
    batch_size:  int = 8
    num_workers: Optional[int] = None   # None → auto-resolved at runtime

    # ---- Training schedule ---------------------------------------------------
    epochs:        int = 60
    warmup_epochs: int = 5
    patience:      int = 12             # early-stopping patience (val macro-F1)

    # ---- Optimiser (AdamW) ---------------------------------------------------
    lr_head:      float = 2e-4
    lr_stage7:    float = 2e-5
    lr_rest:      float = 5e-6
    weight_decay: float = 1e-4

    # ---- Regularisation ------------------------------------------------------
    label_smoothing: float = 0.1
    dropout:         float = 0.4
    mixup_alpha:     float = 0.3        # 0 disables Mixup
    grad_clip:       float = 1.0        # max gradient norm (L2)

    # ---- XAI -----------------------------------------------------------------
    gradcam_samples: int = 40

    # ---- Reproducibility -----------------------------------------------------
    seed: int = 42

    # ---- MedGemma (optional) -------------------------------------------------
    enable_medgemma:          bool = False
    medgemma_model_id:        str  = "google/medgemma-4b-it"
    medgemma_max_samples:     int  = 15
    medgemma_max_new_tokens:  int  = 512

    # ---- Resume --------------------------------------------------------------
    resume: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config to a plain dict (e.g. for JSON logging)."""
        return asdict(self)

    def __repr__(self) -> str:
        lines = [f"TrainingConfig("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# ==============================================================================
# CONFIG LOADER
# ==============================================================================

class ConfigLoader:
    """
    Builds a ``TrainingConfig`` by merging (in priority order):

    1. Dataclass defaults
    2. YAML file (if provided via ``--config``)
    3. CLI argument overrides

    Example
    -------
    >>> loader = ConfigLoader()
    >>> cfg = loader.load()
    >>> print(cfg.lr_head)
    0.0002
    """

    def __init__(self):
        self._parser = self._build_parser()

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def load(self) -> TrainingConfig:
        """Parse CLI + optional YAML and return a ``TrainingConfig``."""
        args = self._parser.parse_args()
        cfg  = TrainingConfig()

        # Layer 2: YAML overrides
        if hasattr(args, "config") and args.config:
            yaml_vals = self._load_yaml(args.config)
            for k, v in yaml_vals.items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)

        # Layer 3: CLI overrides (only when explicitly set)
        for k, v in vars(args).items():
            if k == "config":
                continue
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(
            description="TRACE ConvNeXt-Base skin-cancer classifier.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        ap.add_argument("--config",      default=None,
                        help="Optional YAML config file path.")
        # Paths
        ap.add_argument("--data_root",   default=None)
        ap.add_argument("--output_dir",  default=None)
        # Data
        ap.add_argument("--image_size",  type=int,   default=None)
        ap.add_argument("--batch_size",  type=int,   default=None)
        ap.add_argument("--num_workers", type=int,   default=None)
        # Schedule
        ap.add_argument("--epochs",         type=int,   default=None)
        ap.add_argument("--warmup_epochs",  type=int,   default=None)
        ap.add_argument("--patience",       type=int,   default=None)
        # LR
        ap.add_argument("--lr_head",    type=float, default=None)
        ap.add_argument("--lr_stage7",  type=float, default=None)
        ap.add_argument("--lr_rest",    type=float, default=None)
        ap.add_argument("--weight_decay", type=float, default=None)
        # Regularisation
        ap.add_argument("--label_smoothing", type=float, default=None)
        ap.add_argument("--dropout",         type=float, default=None)
        ap.add_argument("--mixup_alpha",     type=float, default=None)
        ap.add_argument("--grad_clip",       type=float, default=None)
        # XAI
        ap.add_argument("--gradcam_samples", type=int, default=None)
        # Misc
        ap.add_argument("--seed",    type=int,   default=None)
        ap.add_argument("--resume",  default=None)
        ap.add_argument("--enable_medgemma", action="store_true", default=None)
        return ap

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        try:
            import yaml
        except ImportError:
            os.system("pip install pyyaml -q")
            import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}