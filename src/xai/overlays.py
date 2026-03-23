# -*- coding: utf-8 -*-
"""
src/xai/overlays.py
====================
YOLO detection overlay visualisation — side-by-side pseudo GT vs prediction.

Classes
-------
OverlaySaver
    Generates paired overlay images (pseudo GT | YOLO prediction) for a
    random sample of test images, computing IoU between GT and predicted box.

Standalone helpers
------------------
xyxy(cx, cy, w, h, W, H)   — normalised box → pixel coords
iou(a, b)                   — axis-aligned IoU between two xyxy boxes
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.utils.io_ops import LiveLogger


# ==============================================================================
# BOX HELPERS
# ==============================================================================

def xyxy(
    cx: float, cy: float, w: float, h: float, W: int, H: int
) -> np.ndarray:
    """Convert normalised (cx, cy, w, h) to pixel (x1, y1, x2, y2)."""
    return np.array([
        (cx - w / 2) * W,
        (cy - h / 2) * H,
        (cx + w / 2) * W,
        (cy + h / 2) * H,
    ], dtype=np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Axis-aligned IoU between two xyxy boxes."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-8))


# ==============================================================================
# OVERLAY SAVER
# ==============================================================================

class OverlaySaver:
    """
    Saves side-by-side overlay images comparing pseudo GT boxes against
    YOLO model predictions.

    Each output figure has two panels:
    - Left  — original image + dashed pseudo GT box (lime)
    - Right — original image + YOLO predicted box (red) + pseudo GT (faded)
              with IoU score in the title

    Parameters
    ----------
    best_pt     : Path         — best YOLO checkpoint (``.pt``)
    gen         : PseudoBoxGenerator — used to get pseudo GT box per image
    out_dir     : Path         — directory for overlay PNGs and JSON report
    device_list : List[int]    — GPU indices (first is used for inference)
    imgsz       : int          — YOLO inference image size
    log         : LiveLogger

    Example
    -------
    >>> saver = OverlaySaver(best_pt, gen, Path("./overlays"), [0], 640, log)
    >>> info = saver.save(test_records, class_names, n=50)
    """

    def __init__(
        self,
        best_pt:     Path,
        gen,                         # PseudoBoxGenerator
        out_dir:     Path,
        device_list: List[int],
        imgsz:       int,
        log:         LiveLogger,
    ):
        self._best_pt     = best_pt
        self._gen         = gen
        self._out_dir     = out_dir
        self._device      = device_list[0] if device_list else 0
        self._imgsz       = imgsz
        self._log         = log

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def save(
        self,
        records:     List[Tuple[Path, str, int]],
        class_names: List[str],
        n:           int = 50,
    ) -> List[Dict]:
        """
        Generate ``n`` overlay images from ``records``.

        Returns
        -------
        List of per-sample info dicts (also written to ``overlay_info.json``).
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            self._log.log("[overlay] skip — ultralytics not available")
            return []

        self._out_dir.mkdir(parents=True, exist_ok=True)
        model   = YOLO(str(self._best_pt))
        samples = random.sample(records, min(n, len(records)))
        info    = []

        for idx, (img_path, true_class, true_idx) in enumerate(samples):
            try:
                result = self._process_one(
                    model, img_path, true_class, true_idx,
                    class_names, idx,
                )
                info.append(result)
            except Exception as e:
                self._log.log(f"[overlay] {idx} failed: {e}")

        (self._out_dir / "overlay_info.json").write_text(
            json.dumps(info, indent=2)
        )
        self._log.log(f"[overlay] {len(info)} overlays → {self._out_dir}")
        return info

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _process_one(
        self,
        model,
        img_path:    Path,
        true_class:  str,
        true_idx:    int,
        class_names: List[str],
        idx:         int,
    ) -> Dict:
        img  = Image.open(img_path).convert("RGB")
        W, H = img.size
        arr  = np.array(img)

        # Pseudo GT box
        cx, cy, bw, bh = self._gen.get_box(img_path)
        gt = xyxy(cx, cy, bw, bh, W, H)

        # YOLO prediction
        res = model.predict(
            source=str(img_path), imgsz=self._imgsz,
            conf=0.01, iou=0.45,
            verbose=False, device=self._device,
        )
        pred_boxes = pred_confs = pred_cls = []
        if res and res[0].boxes is not None and len(res[0].boxes):
            pred_boxes = res[0].boxes.xyxy.cpu().numpy()
            pred_confs = res[0].boxes.conf.cpu().numpy().tolist()
            pred_cls   = res[0].boxes.cls.cpu().numpy().astype(int).tolist()

        # Best predicted box = highest IoU with GT
        best_iou = 0.0
        best_box: Optional[np.ndarray] = None
        best_conf = 0.0
        best_cls  = true_idx
        for pb, pc, pl in zip(pred_boxes, pred_confs, pred_cls):
            iv = iou(gt, pb)
            if iv > best_iou:
                best_iou = iv; best_box = pb
                best_conf = pc; best_cls = pl

        # --- Figure ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(arr)
        axes[0].add_patch(mpatches.Rectangle(
            (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
            lw=2.5, edgecolor="lime", facecolor="none",
            ls="--", label="Pseudo GT",
        ))
        axes[0].set_title(f"True: {true_class}", fontsize=10)
        axes[0].legend(fontsize=8); axes[0].axis("off")

        axes[1].imshow(arr)
        if best_box is not None:
            pred_name = (class_names[best_cls]
                         if best_cls < len(class_names) else str(best_cls))
            axes[1].add_patch(mpatches.Rectangle(
                (best_box[0], best_box[1]),
                best_box[2] - best_box[0], best_box[3] - best_box[1],
                lw=2.5, edgecolor="red", facecolor="none",
                label=f"Pred:{pred_name}({best_conf:.2f})",
            ))
            axes[1].add_patch(mpatches.Rectangle(
                (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                lw=1.5, edgecolor="lime", facecolor="none",
                ls="--", alpha=0.5,
            ))
            axes[1].legend(fontsize=8)
        axes[1].set_title(f"IoU={best_iou:.3f}", fontsize=10)
        axes[1].axis("off")

        pred_name_final = (
            class_names[best_cls] if best_cls < len(class_names) else "?"
        )
        plt.suptitle(
            f"True:{true_class}  Pred:{pred_name_final}  "
            f"Conf:{best_conf:.2f}  IoU:{best_iou:.3f}",
            fontsize=9,
        )
        plt.tight_layout()

        fname = f"overlay_{idx:03d}.png"
        plt.savefig(self._out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "sample":     idx,
            "image":      str(img_path),
            "true_class": true_class,
            "pred_class": pred_name_final,
            "pred_conf":  round(best_conf, 4),
            "iou":        round(best_iou, 4),
            "gt_box":     [round(float(v), 4) for v in gt],
            "pred_box":   ([round(float(v), 4) for v in best_box]
                           if best_box is not None else None),
            "overlay":    fname,
            "box_norm":   {
                "cx": round(cx, 4), "cy": round(cy, 4),
                "w":  round(bw, 4), "h":  round(bh, 4),
            },
        }

    def __repr__(self) -> str:
        return (
            f"OverlaySaver("
            f"out_dir={self._out_dir}, "
            f"imgsz={self._imgsz})"
        )