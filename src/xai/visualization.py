# -*- coding: utf-8 -*-
"""
src/xai/visualization.py
=========================
Training curve and evaluation visualisation for ConvNeXt-Base.

Classes
-------
TrainingPlotter
    Saves all standard training plots: loss/accuracy curves, macro-F1,
    generalisation gap, LR schedule, balanced accuracy, confusion matrices,
    ROC curves, PR curves, per-class P/R/F1, and confidence distributions.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
)

from src.utils.io_ops import LiveLogger


class TrainingPlotter:
    """
    Saves all training and evaluation visualisation plots.

    Parameters
    ----------
    out_dir : Path — root output directory; plots written to ``out_dir/plots/``
    log     : LiveLogger

    Example
    -------
    >>> plotter = TrainingPlotter(Path("./convNext"), log)
    >>> plotter.save_training_curves(history)
    >>> plotter.save_split_plots(split_metrics, class_names)
    """

    def __init__(self, out_dir: Path, log: LiveLogger):
        self._out_dir   = out_dir
        self._plots_dir = out_dir / "plots"
        self._log       = log
        self._plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def save_all(self, history: Dict, split_metrics: Dict, class_names: List[str]) -> None:
        """Convenience wrapper: save all training + evaluation plots."""
        self.save_training_curves(history)
        self.save_split_plots(split_metrics, class_names)

    def save_training_curves(self, history: Dict) -> None:
        """Save loss, accuracy, macro-F1, generalisation gap, LR, balanced-acc plots."""
        ep = list(range(1, len(history["train_loss"]) + 1))
        self._plot_loss_acc(ep, history)
        self._plot_macro_f1(ep, history)
        self._plot_gen_gap(ep, history)
        self._plot_lr(ep, history)
        self._plot_balanced_acc(ep, history)

    def save_split_plots(self, split_metrics: Dict, class_names: List[str]) -> None:
        """Save per-split evaluation plots: CM, ROC, PR, P/R/F1, confidence dist."""
        for split_name, sm in split_metrics.items():
            y_true = np.array(sm["labels"])
            y_pred = np.array(sm["preds"])
            y_prob = np.array(sm["probs"])
            nc     = len(class_names)
            yoh    = np.eye(nc)[y_true.astype(int)]
            colors = plt.cm.tab10(np.linspace(0, 1, nc))

            self._plot_confusion_counts(y_true, y_pred, class_names, split_name)
            self._plot_confusion_norm(y_true, y_pred, class_names, split_name)
            self._plot_roc(yoh, y_prob, class_names, colors, split_name)
            self._plot_pr(yoh, y_prob, class_names, colors, split_name)
            self._plot_per_class_prf1(y_true, y_pred, class_names, split_name)
            self._plot_confidence_dist(y_true, y_pred, y_prob, class_names, split_name)

    # ------------------------------------------------------------------
    # PRIVATE — training curves
    # ------------------------------------------------------------------

    def _save(self, name: str) -> None:
        plt.tight_layout()
        plt.savefig(self._plots_dir / name, dpi=150, bbox_inches="tight")
        plt.close()
        self._log.log(f"  [plot] {name}")

    def _plot_loss_acc(self, ep, history):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("ConvNeXt-Base | Training Curves", fontweight="bold")
        axes[0].plot(ep, history["train_loss"], label="Train")
        axes[0].plot(ep, history["val_loss"],   label="Val")
        axes[0].set(title="Loss", xlabel="Epoch")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep, history["train_acc"], label="Train")
        axes[1].plot(ep, history["val_acc"],   label="Val")
        axes[1].axhline(max(history["val_acc"]), color="green", ls="--", lw=1.2,
                        label=f"Best={max(history['val_acc']):.4f}")
        axes[1].set(title="Accuracy", xlabel="Epoch")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        self._save("01_loss_accuracy.png")

    def _plot_macro_f1(self, ep, history):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ep, history["train_macro_f1"], label="Train")
        ax.plot(ep, history["val_macro_f1"],   label="Val")
        ax.set(title="Macro-F1", xlabel="Epoch")
        ax.legend(); ax.grid(alpha=0.3)
        self._save("02_macro_f1.png")

    def _plot_gen_gap(self, ep, history):
        fig, ax = plt.subplots(figsize=(12, 4))
        gap_acc = np.array(history["train_acc"]) - np.array(history["val_acc"])
        gap_f1  = np.array(history["train_macro_f1"]) - np.array(history["val_macro_f1"])
        ax.plot(ep, gap_acc, label="Acc gap", color="blue")
        ax.plot(ep, gap_f1,  label="F1 gap",  color="orange")
        ax.axhline(0,    color="gray", lw=1, ls="--")
        ax.axhline(0.08, color="red",  lw=1, ls="--", label="Overfit threshold (0.08)")
        ax.fill_between(ep, gap_acc, 0,
                        where=np.array(gap_acc) > 0.08,
                        alpha=0.15, color="red", label="Overfit zone")
        ax.set(title="Generalisation Gap (Train - Val)", xlabel="Epoch")
        ax.legend(); ax.grid(alpha=0.3)
        self._save("03_generalisation_gap.png")

    def _plot_lr(self, ep, history):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ep, history["lr_head"], color="#7c3aed", lw=2)
        ax.set(title="LR Schedule (head group)", xlabel="Epoch")
        ax.set_yscale("log"); ax.grid(alpha=0.3)
        self._save("04_lr_schedule.png")

    def _plot_balanced_acc(self, ep, history):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ep, history["train_balanced_acc"], label="Train")
        ax.plot(ep, history["val_balanced_acc"],   label="Val")
        ax.set(title="Balanced Accuracy", xlabel="Epoch")
        ax.legend(); ax.grid(alpha=0.3)
        self._save("05_balanced_accuracy.png")

    # ------------------------------------------------------------------
    # PRIVATE — per-split plots
    # ------------------------------------------------------------------

    def _plot_confusion_counts(self, y_true, y_pred, class_names, pref):
        nc  = len(class_names)
        cm  = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(7, nc), max(6, nc - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, annot_kws={"size": 10})
        ax.set(title=f"Confusion Matrix (counts) — {pref}",
               xlabel="Predicted", ylabel="True")
        self._save(f"{pref}_cm_counts.png")

    def _plot_confusion_norm(self, y_true, y_pred, class_names, pref):
        nc   = len(class_names)
        cm   = confusion_matrix(y_true, y_pred)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(max(7, nc), max(6, nc - 1)))
        sns.heatmap(cm_n, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10})
        ax.set(title=f"Normalised Confusion Matrix — {pref}",
               xlabel="Predicted", ylabel="True")
        self._save(f"{pref}_cm_normalised.png")

    def _plot_roc(self, yoh, y_prob, class_names, colors, pref):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (cn, c) in enumerate(zip(class_names, colors)):
                fpr, tpr, _ = roc_curve(yoh[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, color=c, lw=2, label=f"{cn} AUC={auc(fpr, tpr):.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set(title=f"ROC Curves — {pref}", xlabel="FPR", ylabel="TPR")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(alpha=0.3)
            self._save(f"{pref}_roc.png")
        except Exception:
            pass

    def _plot_pr(self, yoh, y_prob, class_names, colors, pref):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (cn, c) in enumerate(zip(class_names, colors)):
                prec_c, rec_c, _ = precision_recall_curve(yoh[:, i], y_prob[:, i])
                ax.plot(rec_c, prec_c, color=c, lw=2,
                        label=f"{cn} AP={auc(rec_c, prec_c):.3f}")
            ax.set(title=f"Precision-Recall — {pref}",
                   xlabel="Recall", ylabel="Precision")
            ax.legend(loc="lower left", fontsize=9)
            ax.grid(alpha=0.3)
            self._save(f"{pref}_pr_curves.png")
        except Exception:
            pass

    def _plot_per_class_prf1(self, y_true, y_pred, class_names, pref):
        nc = len(class_names)
        rd = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
        fig, ax = plt.subplots(figsize=(max(10, nc * 2), 5))
        x = np.arange(nc)
        w = 0.25
        for k, (key, lbl, col) in enumerate([
            ("precision", "Precision", "#2563eb"),
            ("recall",    "Recall",    "#16a34a"),
            ("f1-score",  "F1",        "#dc2626"),
        ]):
            vals = [rd[cn][key] for cn in class_names]
            bars = ax.bar(x + k * w, vals, w, label=lbl, color=col, alpha=0.85)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2.0, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=40)
        ax.set_xticks(x + w)
        ax.set_xticklabels(class_names, rotation=15, ha="right")
        ax.set(title=f"Per-Class P/R/F1 — {pref}", ylim=(0, 1.15))
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        self._save(f"{pref}_per_class_metrics.png")

    def _plot_confidence_dist(self, y_true, y_pred, y_prob, class_names, pref):
        nc = len(class_names)
        fig, axes2 = plt.subplots(1, nc, figsize=(4 * nc, 4), sharey=True)
        if nc == 1:
            axes2 = [axes2]
        for i, (cn, ax2) in enumerate(zip(class_names, axes2)):
            mask   = y_true == i
            c_conf = y_prob[mask & (y_pred == y_true), i]
            w_conf = y_prob[mask & (y_pred != y_true), i]
            ax2.hist(c_conf, bins=20, alpha=0.7, color="green",
                     label="Correct",   density=True)
            ax2.hist(w_conf, bins=20, alpha=0.7, color="red",
                     label="Incorrect", density=True)
            ax2.set(title=cn, xlabel="Confidence")
            ax2.legend(fontsize=7)
        fig.suptitle(
            f"Confidence Distribution per Class — {pref}", fontweight="bold"
        )
        self._save(f"{pref}_confidence_dist.png")

    def __repr__(self) -> str:
        return f"TrainingPlotter(out_dir={self._out_dir})"
