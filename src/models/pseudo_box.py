# -*- coding: utf-8 -*-
"""
src/models/pseudo_box.py
========================
Pseudo bounding-box generation for YOLO training labels.

Since no manual bounding-box annotations exist, pseudo-boxes are derived from:

1. **locmap mode** — uses the exp7 classification model's localisation map
   (more accurate when the exp7 checkpoint is available).
2. **otsu mode**   — dermoscopy-aware Otsu thresholding fallback
   (used when no exp7 checkpoint is found).

Classes
-------
PseudoBoxGenerator
    Main entry point.  Call :meth:`get_box` per image.

Standalone helpers
------------------
otsu_box(img_path)      — Otsu-based box for a single image
locmap_box(lm, thresh)  — Activation-map-based box
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils.io_ops import LiveLogger


# ==============================================================================
# LOW-LEVEL BOX HELPERS
# ==============================================================================

def _clamp(
    cx: float, cy: float, bw: float, bh: float, mn: float = 0.08
) -> Tuple[float, float, float, float]:
    """Clamp box to valid normalised coordinates."""
    bw = float(np.clip(bw, mn, 0.95))
    bh = float(np.clip(bh, mn, 0.95))
    cx = float(np.clip(cx, bw / 2 + 0.01, 1 - bw / 2 - 0.01))
    cy = float(np.clip(cy, bh / 2 + 0.01, 1 - bh / 2 - 0.01))
    return cx, cy, bw, bh


def _mask_to_box(
    mask: np.ndarray, H: int, W: int, pad: float = 0.04
) -> Tuple[float, float, float, float]:
    """Convert a binary mask to a normalised (cx, cy, bw, bh) box."""
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    if not len(rows) or not len(cols):
        return 0.5, 0.5, 0.40, 0.40

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    pr = max(1, int((r1 - r0) * pad))
    pc = max(1, int((c1 - c0) * pad))
    r0 = max(0, r0 - pr);  r1 = min(H - 1, r1 + pr)
    c0 = max(0, c0 - pc);  c1 = min(W - 1, c1 + pc)

    return _clamp(
        (c0 + c1) / 2.0 / W,
        (r0 + r1) / 2.0 / H,
        (c1 - c0) / float(W),
        (r1 - r0) / float(H),
    )


# ==============================================================================
# OTSU BOX
# ==============================================================================

def otsu_box(img_path: Path) -> Tuple[float, float, float, float]:
    """
    Dermoscopy-aware Otsu pseudo-box.

    Handles:
    - Vignette / dark corners (common in dermoscopy hardware)
    - Both dark-lesion-on-light-skin AND light-lesion-on-dark-skin
    - Selects best candidate by centrality + size scoring

    Fallback: ``(0.5, 0.5, 0.40, 0.40)`` — safe central crop.
    The fallback box covers ~16% of the image, chosen to avoid the
    NaN cls_loss explosion caused by an oversized 0.75 default box
    that covered 56% of the image.
    """
    try:
        from scipy.ndimage import label as scipy_label, binary_fill_holes

        img  = Image.open(img_path).convert("RGB")
        arr  = np.array(img, np.float32)
        H, W = arr.shape[:2]

        # Green channel has the best lesion contrast in dermoscopy;
        # luminance as a second candidate
        green = arr[:, :, 1]
        lum   = (0.299 * arr[:, :, 0]
                 + 0.587 * arr[:, :, 1]
                 + 0.114 * arr[:, :, 2])

        best_box   = None
        best_score = -1.0

        for channel in [green, lum]:
            # ---------- Otsu threshold ----------
            c   = np.clip(channel.flatten().astype(np.int32), 0, 255)
            h   = np.bincount(c, minlength=256).astype(np.float64)
            tot = float(c.size)
            bt  = 128; bv = -1.0; w0 = s0 = 0.0
            ts  = float(np.sum(np.arange(256) * h))

            for t in range(1, 256):
                w0 += h[t - 1] / tot
                w1  = 1.0 - w0
                if w0 < 1e-10 or w1 < 1e-10:
                    continue
                s0  += (t - 1) * h[t - 1] / tot
                mu0  = s0 / w0
                mu1  = (ts / tot - s0) / w1
                v    = w0 * w1 * (mu0 - mu1) ** 2
                if v > bv:
                    bv = v; bt = t

            for dark_lesion in [False, True]:
                mask = (
                    (channel < bt if dark_lesion else channel >= bt)
                    .astype(np.uint8)
                )

                # Zero out 5% border — removes vignette ring
                bh_ = max(1, int(H * 0.05))
                bw_ = max(1, int(W * 0.05))
                mask[:bh_, :]  = 0;  mask[-bh_:, :] = 0
                mask[:, :bw_]  = 0;  mask[:, -bw_:] = 0

                # Fill holes (specular reflections inside lesion)
                try:
                    mask = binary_fill_holes(mask).astype(np.uint8)
                except Exception:
                    pass

                frac = mask.mean()
                # Valid range: 5% to 70% of image area
                if frac < 0.05 or frac > 0.70:
                    continue

                lb, nc = scipy_label(mask)
                if nc == 0:
                    continue

                sizes       = np.bincount(lb.flatten())
                sizes[0]    = 0
                comp        = (lb == int(np.argmax(sizes))).astype(np.uint8)
                rows        = np.where(comp.any(1))[0]
                cols        = np.where(comp.any(0))[0]
                if not len(rows) or not len(cols):
                    continue

                cy_f       = (rows.min() + rows.max()) / 2.0 / H
                cx_f       = (cols.min() + cols.max()) / 2.0 / W
                # Centrality: 1.0 at image centre, 0.0 at corner
                centrality  = 1.0 - 2.0 * max(abs(cx_f - 0.5), abs(cy_f - 0.5))
                # Size score: peak at 25% of image area (typical lesion)
                size_score  = 1.0 - abs(comp.mean() - 0.25) / 0.25
                score       = centrality * 0.6 + max(0.0, size_score) * 0.4

                if score > best_score:
                    best_score = score
                    best_box   = _mask_to_box(comp, H, W)

        return best_box if best_box is not None else (0.5, 0.5, 0.40, 0.40)

    except Exception:
        return 0.5, 0.5, 0.40, 0.40


# ==============================================================================
# LOCMAP BOX
# ==============================================================================

def locmap_box(
    lm: np.ndarray, thresh: float = 0.35
) -> Tuple[float, float, float, float]:
    """
    Convert a localisation activation map to a normalised bounding box.

    Parameters
    ----------
    lm     : np.ndarray (H, W) — raw activation map from exp7 model
    thresh : float              — activation threshold (default 0.35)
    """
    h, w = lm.shape
    n    = lm - lm.min()
    d    = n.max()
    if d < 1e-8:
        return 0.5, 0.5, 0.40, 0.40
    n /= d
    mask = (n >= thresh).astype(np.uint8)
    if mask.sum() < 9:
        mask = (n >= thresh * 0.5).astype(np.uint8)
    if not mask.sum():
        return 0.5, 0.5, 0.40, 0.40
    return _mask_to_box(mask, h, w)


# ==============================================================================
# PSEUDO BOX GENERATOR
# ==============================================================================

class PseudoBoxGenerator:
    """
    Generates pseudo bounding boxes for YOLO training.

    Mode selection
    --------------
    ``locmap`` — uses exp7 classification model's localisation map
                 (preferred when exp7 checkpoint is available).
    ``otsu``   — falls back to dermoscopy-aware Otsu thresholding.

    Parameters
    ----------
    exp7_dir : Path or None — directory containing exp7 ``best_model.pth``
    device   : torch.device
    logger   : LiveLogger
    thresh   : float        — activation threshold for locmap mode

    Example
    -------
    >>> gen = PseudoBoxGenerator(exp7_dir=Path("./exp7"), device=device, logger=log)
    >>> cx, cy, bw, bh = gen.get_box(Path("lesion.jpg"))
    """

    def __init__(
        self,
        exp7_dir: Optional[Path],
        device:   torch.device,
        logger:   LiveLogger,
        thresh:   float = 0.35,
    ):
        self._device = device
        self._logger = logger
        self._thresh = thresh
        self._model:  Optional[nn.Module] = None
        self._tfm     = None
        self._mode    = "otsu"

        if exp7_dir is None:
            logger.log("[pseudo] No exp7 dir — using Otsu")
            return

        pth = self._find_checkpoint(Path(exp7_dir))
        if pth is None:
            logger.log("[pseudo] No checkpoint found — using Otsu")
            return

        logger.log(f"[pseudo] Loading exp7 checkpoint: {pth}")
        try:
            ck   = torch.load(pth, map_location=device)
            nc   = len(ck.get("names", []) or []) + 1 or 9
            self._model = self._build_proxy(nc).to(device).eval()
            st   = {k.replace("module.", ""): v
                    for k, v in ck["state"].items()}
            miss, _ = self._model.load_state_dict(st, strict=False)
            logger.log(f"[pseudo] Loaded. Missing keys={len(miss)}")
            self._build_transform()
            self._mode = "locmap"
        except Exception as e:
            logger.log(f"[pseudo] Load failed ({e}) — using Otsu")
            self._model = None

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_box(self, p: Path) -> Tuple[float, float, float, float]:
        """Return a normalised (cx, cy, bw, bh) pseudo bounding box."""
        if self._mode == "locmap" and self._model is not None:
            try:
                img = Image.open(p).convert("RGB")
                t   = self._tfm(img).unsqueeze(0).to(self._device)
                out = self._model(t)
                lm  = out.get("loc_map") if isinstance(out, dict) else None
                if lm is not None:
                    return locmap_box(lm.squeeze().cpu().numpy(), self._thresh)
            except Exception:
                pass
        return otsu_box(p)

    @property
    def mode(self) -> str:
        """Active mode: ``'locmap'`` or ``'otsu'``."""
        return self._mode

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _find_checkpoint(root: Path) -> Optional[Path]:
        candidates = list(root.rglob("best_model.pth"))
        return (max(candidates, key=lambda p: p.stat().st_mtime)
                if candidates else None)

    @staticmethod
    def _build_proxy(nc: int) -> nn.Module:
        """Lightweight ResNet-50 proxy that mimics the exp7 model interface."""
        from torchvision.models import resnet50

        class _Proxy(nn.Module):
            def __init__(self, nc_: int):
                super().__init__()
                b         = resnet50(weights=None)
                self.body = nn.Sequential(*list(b.children())[:-2])
                self.loc  = nn.Sequential(
                    nn.Conv2d(2048, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.SiLU(),
                    nn.Conv2d(256, 1, 1),
                    nn.Sigmoid(),
                )
                self.gap  = nn.AdaptiveAvgPool2d(1)
                self.fc   = nn.Linear(2048, nc_)

            def forward(self, x):
                f = self.body(x)
                return {
                    "logits":  self.fc(self.gap(f).flatten(1)),
                    "loc_map": self.loc(f),
                }

        return _Proxy(nc)

    def _build_transform(self) -> None:
        from torchvision import transforms
        self._tfm = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    def __repr__(self) -> str:
        return f"PseudoBoxGenerator(mode={self._mode}, thresh={self._thresh})"