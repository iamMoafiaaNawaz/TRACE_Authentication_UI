# -*- coding: utf-8 -*-
"""
evaluation/yolo_plots.py
=========================
Summary visualisations for the YOLO localisation pipeline.

Classes
-------
YoloPlotter
    Generates bar-chart summaries (precision, recall, F1, mAP@0.5,
    mAP@0.5:0.95) across all trained YOLO variants, plus a per-class
    AP bar chart for the best-performing model.

Usage
-----
>>> plotter = YoloPlotter()
>>> plotter.save(all_results, class_names, Path("./exp8/plots"), log)

``all_results`` shape (produced by YoloEvaluator.evaluate_all)::

    {
        "yolov11x": {
            "val":  {"precision": 0.82, "recall": 0.79, "f1": 0.80,
                     "mAP_50": 0.81, "mAP_50_95": 0.53, "per_class_AP": {...}},
            "test": {...},
        },
        "yolov10x": {...},
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io_ops import LiveLogger


# ==============================================================================
# YOLO PLOTTER
# ==============================================================================

class YoloPlotter:
    """
    Generates comparative bar charts for multiple YOLO variants.

    Parameters
    ----------
    dpi : int
        Resolution for saved figures (default 200).

    Example
    -------
    >>> YoloPlotter().save(all_results, class_names, out_dir, log)
    """

    _METRICS = [
        ("mAP_50",     "mAP@0.5"),
        ("mAP_50_95",  "mAP@0.5:0.95"),
        ("precision",  "Precision"),
        ("recall",     "Recall"),
        ("f1",         "F1"),
    ]

    def __init__(self, dpi: int = 200):
        self._dpi = dpi

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def save(
        self,
        all_results: Dict[str, Dict],
        class_names: List[str],
        out_dir: Path,
        log: LiveLogger,
    ) -> None:
        """
        Save all summary plots to ``out_dir``.

        Generates one bar chart per metric (val + test bars side by side)
        and one per-class AP chart for the best model on test set.

        Parameters
        ----------
        all_results : dict — variant → {split → metrics dict}
        class_names : list of str
        out_dir     : Path — directory where PNGs are written
        log         : LiveLogger
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_names = list(all_results.keys())
        if not model_names:
            log.log("[plots] No results to plot.")
            return

        for mk, ml in self._METRICS:
            self._save_metric_chart(all_results, model_names, mk, ml, out_dir)

        self._save_per_class_ap(all_results, model_names, class_names, out_dir, log)
        log.log(f"[plots] Saved {len(self._METRICS) + 1} charts -> {out_dir}")

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _save_metric_chart(
        self,
        all_results: Dict,
        model_names: List[str],
        metric_key: str,
        metric_label: str,
        out_dir: Path,
    ) -> None:
        """Bar chart: model_names × (val, test) for a single metric."""
        x   = np.arange(len(model_names))
        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 5))

        for si, split in enumerate(["val", "test"]):
            vals = [
                all_results[m].get(split, {}).get(metric_key, 0.0)
                for m in model_names
            ]
            bars = ax.bar(x + (si - 0.5) * 0.35, vals, 0.35, label=split.capitalize())
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    v + 0.005,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"summary_{metric_key}.png", dpi=self._dpi)
        plt.close()

    def _save_per_class_ap(
        self,
        all_results: Dict,
        model_names: List[str],
        class_names: List[str],
        out_dir: Path,
        log: LiveLogger,
    ) -> None:
        """Per-class AP bar chart for the best model (by test mAP@0.5:0.95)."""
        best_model = max(
            model_names,
            key=lambda m: all_results[m].get("test", {}).get("mAP_50_95", 0.0),
        )
        per_class = all_results[best_model].get("test", {}).get("per_class_AP", {})
        if not per_class:
            log.log("[plots] No per-class AP data available for best model.")
            return

        cn   = list(per_class.keys())
        vals = [per_class[c] for c in cn]

        fig, ax = plt.subplots(figsize=(max(8, len(cn) * 1.4), 5))
        bars = ax.bar(cn, vals, color="steelblue")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_ylim(0, 1.08)
        ax.set_ylabel("AP@0.5:0.95")
        ax.set_title(f"Per-class AP — {best_model} (best model, test set)")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "per_class_ap_best_model.png", dpi=self._dpi)
        plt.close()
