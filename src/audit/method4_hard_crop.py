# -*- coding: utf-8 -*-
"""
audit/method4_hard_crop.py
===========================
Method 4: Hard Crop Strict Validation Probe

Loads an existing checkpoint and evaluates on 5 deliberately hostile
transforms that progressively strip background, colour, and spatial cues.

A large accuracy drop under stripping = model learned shortcuts (skin-tone,
background, spatial layout) rather than true lesion morphology.

Transform conditions
--------------------
A. Standard          — same as training (baseline)
B. Hard crop 70%     — strips border/background context
C. Greyscale         — removes skin-tone shortcuts
D. Heavy augmentation — strips spatial layout shortcuts
E. Combined          — all of the above simultaneously

FYP defence talking point
--------------------------
"We applied five hostile evaluation conditions that progressively strip
non-lesion features. A performance drop >10% would indicate the model
relies on background or colour shortcuts. Our results show < X% drop,
confirming the model learned lesion morphology."
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from audit.audit_logger import AuditLogger


class HardCropProbe:
    """
    Evaluates a trained ConvNeXt-Base checkpoint under hostile transforms
    to detect shortcut learning.

    Parameters
    ----------
    checkpoint_path : str  — path to ``.pth`` checkpoint
    test_dir        : Path — test split directory (ImageFolder layout)
    log             : AuditLogger
    image_size      : int  (default 512)
    batch_size      : int
    num_workers     : int

    Example
    -------
    >>> probe = HardCropProbe("./best.pth", Path("./data/test"), log)
    >>> result = probe.run(output_dir=Path("./audit"))
    """

    def __init__(
        self,
        checkpoint_path: str,
        test_dir:        Path,
        log:             AuditLogger,
        image_size:      int = 512,
        batch_size:      int = 8,
        num_workers:     int = 4,
    ):
        self._ckpt        = checkpoint_path
        self._test_dir    = test_dir
        self._log         = log
        self._image_size  = image_size
        self._batch_size  = batch_size
        self._num_workers = num_workers

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(self, output_dir: Optional[Path] = None) -> Dict:
        """
        Run all five evaluation conditions and return the results dict.

        Returns ``{"method": "hard_crop_probe", "status": "skipped_no_checkpoint"}``
        if the checkpoint is missing.
        """
        self._log.sep(True)
        self._log.log("METHOD 4: HARD CROP STRICT VALIDATION PROBE")
        self._log.log(
            "  Evaluates checkpoint under feature-stripping transforms."
        )
        self._log.log(
            "  Large accuracy drop → model relies on shortcuts, not morphology."
        )
        self._log.sep()

        if not self._ckpt or not Path(self._ckpt).exists():
            self._log.log(f"  [SKIP] Checkpoint not found: {self._ckpt}")
            return {"method": "hard_crop_probe", "status": "skipped_no_checkpoint"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log.log(f"  Device: {device}")

        model, class_names = self._load_model(device)
        crop_size = int(self._image_size * 0.70)

        # ---- Five evaluation conditions ----
        conditions = {
            "A_Standard":   self._make_tf_standard(),
            "B_HardCrop70": self._make_tf_hard_crop(crop_size),
            "C_Greyscale":  self._make_tf_greyscale(),
            "D_HeavyAug":   self._make_tf_heavy_aug(),
            "E_Combined":   self._make_tf_combined(crop_size),
        }
        labels = {
            "A_Standard":   "Standard (baseline)",
            "B_HardCrop70": "Hard crop 70% (strips background)",
            "C_Greyscale":  "Greyscale (strips skin-tone)",
            "D_HeavyAug":   "Heavy augmentation (strips spatial layout)",
            "E_Combined":   "Combined worst-case",
        }

        results = {}
        for key, tf in conditions.items():
            label = labels[key]
            self._log.log(f"\n  [{key[0]}] {label}:")
            m = self._eval(model, tf, device)
            results[key] = m
            self._log.log(
                f"      acc={m['acc']:.5f}  bal={m['balanced_acc']:.5f}  "
                f"F1={m['macro_f1']:.5f}  MCC={m['mcc']:.5f}"
            )

        # ---- Interpret drops ----
        baseline = results["A_Standard"]["acc"]
        drops = {
            "crop":     baseline - results["B_HardCrop70"]["acc"],
            "grey":     baseline - results["C_Greyscale"]["acc"],
            "aug":      baseline - results["D_HeavyAug"]["acc"],
            "combined": baseline - results["E_Combined"]["acc"],
        }

        self._log.log("\n  INTERPRETATION:")
        self._log.sep()
        for label, drop in [
            ("Background removal (crop)", drops["crop"]),
            ("Colour removal (grey)",     drops["grey"]),
            ("Heavy augmentation",        drops["aug"]),
            ("Combined worst-case",       drops["combined"]),
        ]:
            self._log.log(f"  {label:35s}: {self._interpret(label, drop)}")

        if drops["crop"] > 0.10 or drops["grey"] > 0.10:
            self._log.log(
                "\n  [CONCERN] Significant performance drop under feature stripping."
            )
        else:
            self._log.log(
                "\n  [PASS] Model is robust to feature stripping — "
                "likely learned lesion morphology."
            )

        if output_dir:
            self._save_plot(results, labels, output_dir)

        clean = lambda d: {k: v for k, v in d.items() if k not in ("labels", "preds")}
        return {
            "method":               "hard_crop_probe",
            **{k: clean(v) for k, v in results.items()},
            "drops_from_standard":  {k: round(v, 5) for k, v in drops.items()},
        }

    # ------------------------------------------------------------------
    # PRIVATE — model loading
    # ------------------------------------------------------------------

    def _load_model(self, device):
        from torchvision.models import convnext_base
        ck          = torch.load(self._ckpt, map_location=device, weights_only=False)
        class_names = ck.get("class_names", None)
        num_classes = len(class_names) if class_names else 4
        dropout_p   = (ck.get("args", {}) or {}).get("dropout", 0.4)

        self._log.log(f"  Classes:     {class_names}")
        self._log.log(
            f"  Best epoch:  {ck.get('best_epoch', ck.get('epoch', '?'))}  |  "
            f"Val F1: {ck.get('best_val_macro_f1', '?')}"
        )

        model = convnext_base(weights=None)
        in_f  = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_f, in_f // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(in_f // 2, num_classes),
        )
        sd = ck.get("model_state_dict", ck)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval(), class_names

    # ------------------------------------------------------------------
    # PRIVATE — evaluation
    # ------------------------------------------------------------------

    def _eval(self, model, tf, device) -> Dict:
        ds     = datasets.ImageFolder(str(self._test_dir), transform=tf)
        loader = DataLoader(
            ds, batch_size=self._batch_size, shuffle=False,
            num_workers=self._num_workers,
            pin_memory=(device.type == "cuda"),
        )
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, leave=False, desc="    eval"):
                preds = torch.argmax(model(inputs.to(device)), 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "acc":          round(float(accuracy_score(y_true, y_pred)), 5),
            "balanced_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 5),
            "macro_f1":     round(float(f1), 5),
            "mcc":          round(float(matthews_corrcoef(y_true, y_pred)), 5),
            "preds":        y_pred.tolist(),
            "labels":       y_true.tolist(),
        }

    # ------------------------------------------------------------------
    # PRIVATE — transforms
    # ------------------------------------------------------------------

    def _make_tf_standard(self):
        return transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(self._image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _make_tf_hard_crop(self, crop_size):
        return transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(crop_size),
            transforms.Resize(self._image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _make_tf_greyscale(self):
        return transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(self._image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _make_tf_heavy_aug(self):
        return transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(self._image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _make_tf_combined(self, crop_size):
        return transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.CenterCrop(crop_size),
            transforms.Resize(self._image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # PRIVATE — plot + interpretation
    # ------------------------------------------------------------------

    def _save_plot(self, results, labels, output_dir: Path) -> None:
        metrics   = ["acc", "balanced_acc", "macro_f1", "mcc"]
        keys      = list(results.keys())
        colours   = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
        fig.suptitle(
            "Hard Crop Probe: Performance vs Evaluation Condition",
            fontweight="bold", fontsize=12,
        )
        baseline  = results["A_Standard"]
        for ax, metric in zip(axes, metrics):
            vals     = [results[k][metric] for k in keys]
            xlabels  = [k.split("_", 1)[1] for k in keys]
            bars     = ax.bar(range(len(keys)), vals, color=colours[:len(keys)], alpha=0.85)
            ax.axhline(
                baseline[metric], color="black", lw=1.5, ls="--", label="Baseline"
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=30,
                )
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(xlabels, fontsize=7, rotation=15, ha="right")
            ax.set_title(metric.replace("_", " ").title(), fontsize=9)
            ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.06))
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = output_dir / "hard_crop_probe.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        self._log.log(f"\n  Comparison plot saved: {out}")

    @staticmethod
    def _interpret(label: str, drop: float) -> str:
        if drop > 0.10:
            return f"SIGNIFICANT DROP ({drop:+.3f}) — shortcut detected"
        elif drop > 0.05:
            return f"MODERATE DROP ({drop:+.3f}) — some shortcut dependence"
        elif drop > 0.01:
            return f"MINOR DROP ({drop:+.3f}) — mostly robust"
        elif drop > -0.02:
            return f"NEGLIGIBLE CHANGE ({drop:+.3f}) — robust"
        else:
            return f"IMPROVEMENT ({drop:+.3f}) — more stable without this cue"

    def __repr__(self) -> str:
        return (
            f"HardCropProbe("
            f"image_size={self._image_size}, "
            f"batch_size={self._batch_size})"
        )