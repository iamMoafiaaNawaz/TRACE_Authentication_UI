# -*- coding: utf-8 -*-
"""
src/models/yolo_registry.py
============================
YOLO model registry — maps variant names to pretrained weight filenames
and provides a weight-file resolver.

Supported variants
------------------
yolov11x, yolov11l  — YOLOv11 (C3k2 + C2PSA backbone)
yolov10x, yolov10l  — YOLOv10 (NMS-free dual-assignment)
yolov9e,  yolov9c   — YOLOv9  (GELAN + PGI)
yolov8x             — YOLOv8  (proven stable baseline)

Example
-------
>>> registry = YoloRegistry()
>>> registry.list_variants()
['yolov10l', 'yolov10x', 'yolov11l', 'yolov11x', 'yolov8x', 'yolov9c', 'yolov9e']
>>> weights_path = registry.resolve("yolov11x", Path("/weights"))
"""

from pathlib import Path
from typing import Dict, List


# ==============================================================================
# DEFAULT MODELS FOR EXPERIMENTS
# ==============================================================================

DEFAULT_MODELS = "yolov11x,yolov10x,yolov9e"


# ==============================================================================
# YOLO REGISTRY
# ==============================================================================

class YoloRegistry:
    """
    Central registry mapping YOLO variant names to weight filenames.

    New variants can be added via :meth:`register` without modifying
    any training script.

    Example
    -------
    >>> r = YoloRegistry()
    >>> path = r.resolve("yolov11x", Path("/home/weights"))
    """

    _DEFAULTS: Dict[str, Dict] = {
        "yolov11x": {
            "weights": "yolo11x.pt",
            "desc":    "YOLOv11x — C3k2+C2PSA, highest accuracy",
        },
        "yolov11l": {
            "weights": "yolo11l.pt",
            "desc":    "YOLOv11l — strong accuracy, faster than x",
        },
        "yolov10x": {
            "weights": "yolov10x.pt",
            "desc":    "YOLOv10x — NMS-free dual assignment",
        },
        "yolov10l": {
            "weights": "yolov10l.pt",
            "desc":    "YOLOv10l — NMS-free efficient large",
        },
        "yolov9e": {
            "weights": "yolov9e.pt",
            "desc":    "YOLOv9e  — GELAN+PGI, best on small data",
        },
        "yolov9c": {
            "weights": "yolov9c.pt",
            "desc":    "YOLOv9c  — GELAN+PGI compact",
        },
        "yolov8x": {
            "weights": "yolov8x.pt",
            "desc":    "YOLOv8x  — proven stable baseline",
        },
    }

    def __init__(self):
        self._registry: Dict[str, Dict] = dict(self._DEFAULTS)

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def register(self, name: str, weights_filename: str, desc: str = "") -> None:
        """Register a new YOLO variant."""
        if name in self._registry:
            raise ValueError(f"Variant '{name}' already registered.")
        self._registry[name] = {"weights": weights_filename, "desc": desc}

    def resolve(self, variant: str, weights_dir: Path) -> Path:
        """
        Return the path to pretrained weights for ``variant``.

        Searches ``weights_dir`` first, then the current directory.

        Raises
        ------
        ValueError  – unknown variant name
        FileNotFoundError – weight file not found in either location
        """
        cfg = self._registry.get(variant)
        if not cfg:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {self.list_variants()}"
            )
        for candidate in [Path(weights_dir) / cfg["weights"],
                          Path(cfg["weights"])]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Weight file '{cfg['weights']}' not found in '{weights_dir}'.\n"
            f"Download with: python -c \"from ultralytics import YOLO; "
            f"YOLO('{cfg['weights']}')\""
        )

    def list_variants(self) -> List[str]:
        """Return sorted list of registered variant names."""
        return sorted(self._registry.keys())

    def describe(self, variant: str) -> str:
        """Return the description string for a variant."""
        return self._registry.get(variant, {}).get("desc", "")

    def parse_models_arg(self, models_str: str) -> List[str]:
        """
        Parse a comma-separated models string (e.g. from CLI ``--models``).
        Silently drops unknown names.

        Example
        -------
        >>> registry.parse_models_arg("yolov11x,yolov10x,yolov9e")
        ['yolov11x', 'yolov10x', 'yolov9e']
        """
        return [m.strip() for m in models_str.split(",")
                if m.strip() in self._registry]

    def __repr__(self) -> str:
        return f"YoloRegistry(variants={self.list_variants()})"