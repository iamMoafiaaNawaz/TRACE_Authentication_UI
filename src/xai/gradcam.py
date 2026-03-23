# -*- coding: utf-8 -*-
"""
src/xai/gradcam.py
====================
GradCAM++ implementation for ConvNeXt-Base skin lesion classification.

Classes
-------
GradCAMPlusPlus
    Computes spatially-corrected GradCAM++ saliency maps via
    forward and backward hooks.

GradCAMSaver
    Generates and saves overlay visualisations for a batch of images.

Functions
---------
get_gradcam_target_layer
    Returns the canonical GradCAM++ target layer for ConvNeXt-Base
    (last depthwise conv in stage-7, block-2).

References
----------
Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for
Deep Convolutional Networks", WACV 2018.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifier import unwrap_model
from src.utils.io_ops import LiveLogger


# ==============================================================================
# GRADCAM++
# ==============================================================================

class GradCAMPlusPlus:
    """
    GradCAM++ — Chattopadhyay et al., 2018.

    Registers forward and backward hooks on a target layer to capture
    activations and gradients, then computes spatially-weighted saliency maps.

    Target for ConvNeXt-Base: ``features[7][2].block[0]``
    (last depthwise 7x7 conv in stage-7, block-2)

    Parameters
    ----------
    model        : nn.Module — the full model (unwrapped if DataParallel)
    target_layer : nn.Module — layer to hook

    Example
    -------
    >>> cam_gen = GradCAMPlusPlus(model, get_gradcam_target_layer(model))
    >>> cam = cam_gen(x, class_idx=2)      # (H, W) numpy array in [0, 1]
    >>> analysis = cam_gen.analyse(cam, "MEL")
    >>> cam_gen.remove()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach())
        )
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach())
        )

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove registered hooks to free memory."""
        self._fh.remove()
        self._bh.remove()

    def __call__(
        self,
        x:         torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM++ saliency map.

        Parameters
        ----------
        x         : (1, C, H, W) input tensor (model input space)
        class_idx : target class index; defaults to argmax if None

        Returns
        -------
        cam : (H, W) float32 ndarray normalised to [0, 1]
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[:, class_idx].sum().backward(retain_graph=False)

        g  = self._grads
        a  = self._acts
        g2 = g ** 2
        g3 = g ** 3
        denom   = 2.0 * g2 + g3 * a.sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha   = g2 / denom
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo + 1e-8)

    def analyse(self, cam: np.ndarray, pred_class: str) -> Dict:
        """
        Summarise a saliency map into interpretable statistics.

        Returns a dict with activation percentages, primary region,
        and a human-readable ``xai_summary`` string.
        """
        h, w = cam.shape
        high = (cam >= 0.70).sum() / cam.size * 100
        mid  = ((cam >= 0.40) & (cam < 0.70)).sum() / cam.size * 100
        ys, xs = np.where(cam >= 0.70)

        if len(ys):
            cy = float(ys.mean()) / h
            cx = float(xs.mean()) / w
            v  = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "central"
            hz = "left"  if cx < 0.4 else "right" if cx > 0.6 else "central"
            region = (
                f"{v}-{hz}"
                if not (v == "central" and hz == "central")
                else "central"
            )
        else:
            region = "diffuse"

        peak = float(cam.max())
        strength = "Strong" if peak > 0.8 else "Moderate" if peak > 0.5 else "Weak"
        return {
            "high_activation_pct": round(high, 2),
            "mid_activation_pct":  round(mid, 2),
            "mean_activation":     round(float(cam.mean()), 4),
            "peak_activation":     round(peak, 4),
            "primary_region":      region,
            "xai_summary": (
                f"GradCAM++ for '{pred_class}': focus on {region} region. "
                f"{high:.1f}% pixels >70% activation. Peak={peak:.3f}. "
                f"{strength} spatial evidence."
            ),
        }

    def __repr__(self) -> str:
        return "GradCAMPlusPlus()"


# ==============================================================================
# TARGET LAYER HELPER
# ==============================================================================

def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the canonical GradCAM++ target layer for ConvNeXt-Base.

    Target: ``features[7][2].block[0]``
    — last depthwise 7x7 conv in stage-7, block-2.

    Safely unwraps ``nn.DataParallel`` before layer access.
    """
    return unwrap_model(model).features[7][2].block[0]


# ==============================================================================
# GRADCAM SAVER
# ==============================================================================

class GradCAMSaver:
    """
    Generates and saves GradCAM++ overlay visualisations for a set of
    images from a DataLoader.

    Each output is a side-by-side figure: original | pred overlay | true overlay.
    A JSON report is also saved to ``out_dir/gradcam_report.json``.

    Parameters
    ----------
    model       : nn.Module
    class_names : list[str]
    out_dir     : Path
    device      : torch.device
    log         : LiveLogger

    Example
    -------
    >>> saver = GradCAMSaver(model, class_names, Path("./gradcam"), device, log)
    >>> reports = saver.run(loader, max_samples=40)
    """

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(
        self,
        model:       nn.Module,
        class_names: List[str],
        out_dir:     Path,
        device:      torch.device,
        log:         LiveLogger,
    ):
        self._model       = model
        self._class_names = class_names
        self._out_dir     = out_dir
        self._device      = device
        self._log         = log

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(self, loader, max_samples: int = 40) -> List[Dict]:
        """Generate overlays for up to ``max_samples`` images."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        cam_gen = GradCAMPlusPlus(self._model, get_gradcam_target_layer(self._model))
        self._model.eval()
        reports = []
        count   = 0

        for images, labels in loader:
            if count >= max_samples:
                break
            images = images.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            with torch.no_grad():
                logits  = self._model(images)
                probs_t = torch.softmax(logits, 1)
                preds_t = torch.argmax(logits, 1)

            for i in range(images.size(0)):
                if count >= max_samples:
                    break
                x         = images[i : i + 1]
                pred_idx  = int(preds_t[i].item())
                true_idx  = int(labels[i].item())
                pred_name = self._class_names[pred_idx]
                true_name = self._class_names[true_idx]
                conf      = float(probs_t[i, pred_idx].item())
                correct   = pred_idx == true_idx

                cam_pred = cam_gen(x.clone(), pred_idx)
                cam_true = cam_gen(x.clone(), true_idx) if true_idx != pred_idx else cam_pred
                rgb = (
                    (images[i].cpu() * self._STD + self._MEAN)
                    .clamp(0, 1).permute(1, 2, 0).numpy()
                )

                self._save_figure(rgb, cam_pred, cam_true, pred_name, true_name,
                                  conf, correct, count)

                xai = cam_gen.analyse(cam_pred, pred_name)
                fname = self._fname(count, pred_name, true_name, correct)
                reports.append({
                    "sample_id":       count,
                    "pred_class":      pred_name,
                    "true_class":      true_name,
                    "pred_confidence": round(conf, 4),
                    "correct":         correct,
                    "class_probs": {
                        self._class_names[k]: round(float(probs_t[i, k].item()), 4)
                        for k in range(len(self._class_names))
                    },
                    "gradcam_image": fname,
                    **xai,
                })
                count += 1

        cam_gen.remove()
        report_path = self._out_dir / "gradcam_report.json"
        report_path.write_text(json.dumps(reports, indent=2))
        self._log.log(f"[gradcam] {count} overlays saved -> {self._out_dir}")
        return reports

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _save_figure(self, rgb, cam_pred, cam_true,
                     pred_name, true_name, conf, correct, idx):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(rgb)
        axes[0].axis("off")
        axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=9)

        axes[1].imshow(rgb)
        axes[1].imshow(cam_pred, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[1].axis("off")
        tick = "OK" if correct else "X"
        axes[1].set_title(
            f"GradCAM++ Pred: {pred_name} [{tick}]\nConf: {conf:.1%}", fontsize=9
        )

        axes[2].imshow(rgb)
        axes[2].imshow(cam_true, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[2].axis("off")
        axes[2].set_title(f"GradCAM++ True: {true_name}", fontsize=9)

        plt.suptitle(
            f"Sample {idx:03d} | {'CORRECT' if correct else 'WRONG'} | Conf={conf:.1%}",
            fontsize=10, fontweight="bold",
            color="green" if correct else "red",
        )
        plt.tight_layout()
        fpath = self._out_dir / self._fname(idx, pred_name, true_name, correct)
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()

    def _fname(self, idx, pred_name, true_name, correct) -> str:
        return f"gradcam_{idx:03d}_{pred_name}_{true_name}_{'OK' if correct else 'X'}.png"

    def __repr__(self) -> str:
        return f"GradCAMSaver(out_dir={self._out_dir})"
