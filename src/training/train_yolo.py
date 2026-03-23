# -*- coding: utf-8 -*-
"""
src/training/train_yolo.py
===========================
``YoloTrainer`` — trains a single YOLO variant with stable hyperparameters
and registers all monitoring callbacks.

Stable hyperparameter rationale
---------------------------------
Root cause of NaN explosion in Exp8 run 1:

  ``warmup_bias_lr = 0.1``  (10× too high)  ← THE main NaN cause
  ``amp = True``             (fp16 amplifies NaN propagation)

Combined with noisy Otsu pseudo-labels, these caused a ``cls_loss`` spike
of 157.5 at epoch 4 that corrupted the checkpoint.

Fixes applied in ``STABLE_TRAIN_DEFAULTS``:
  ``warmup_bias_lr = 0.01`` (10× reduction)
  ``amp            = False`` (full fp32)
  ``lr0            = 3e-4``  (3× reduction from 1e-3)
  ``box            = 5.0``   (lower gradient magnitude; was 7.5)
  ``perspective    = 0.0``   (was 0.0005 — causes NaN on extreme crops)
"""

import shutil
import time
import traceback
from pathlib import Path
from typing import List

from src.training.yolo_callbacks import (
    make_end_callback,
    make_epoch_callback,
    make_nan_guard_callback,
    make_start_callback,
)
from src.utils.io_ops import LiveLogger


# ==============================================================================
# STABLE HYPERPARAMETER DEFAULTS
# ==============================================================================

STABLE_TRAIN_DEFAULTS = dict(
    optimizer       = "AdamW",
    lr0             = 3e-4,       # was 1e-3  — 3× reduction
    lrf             = 0.01,
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 3,
    warmup_bias_lr  = 0.01,       # was 0.1   — THE main NaN cause; 10× reduction
    warmup_momentum = 0.8,
    box             = 5.0,        # was 7.5   — lower gradient magnitude
    cls             = 0.5,
    dfl             = 1.5,
    amp             = False,      # was True  — fp16 amplifies NaN
    mosaic          = 0.8,
    mixup           = 0.1,        # was 0.2
    degrees         = 15.0,       # was 30.0
    flipud          = 0.5,
    fliplr          = 0.5,
    hsv_h           = 0.015,
    hsv_s           = 0.4,        # was 0.5
    hsv_v           = 0.3,        # was 0.4
    scale           = 0.4,
    shear           = 2.0,        # was 5.0
    perspective     = 0.0,        # was 0.0005 — causes NaN on extreme crops
    translate       = 0.1,
    dropout         = 0.05,       # was 0.1
    close_mosaic    = 10,
    plots           = False,
    save            = True,
    save_period     = 10,
    verbose         = True,
    seed            = 42,
    deterministic   = True,
    # label_smoothing intentionally omitted — deprecated in ultralytics,
    # harmful for nc=4 (pushes targets away from {0,1} unnecessarily)
)


# ==============================================================================
# YOLO TRAINER
# ==============================================================================

class YoloTrainer:
    """
    Trains a single YOLO variant with stable hyperparameters.

    Registers four callbacks:
    - ``on_train_start``       → start banner
    - ``on_train_epoch_end``   → epoch metrics
    - ``on_train_epoch_end``   → NaN guard
    - ``on_train_end``         → completion summary

    Parameters
    ----------
    variant      : str         — e.g. ``"yolov11x"``
    weights_path : Path        — pretrained ``.pt`` file
    yaml_path    : Path        — YOLO ``dataset.yaml``
    out_dir      : Path        — experiment output root
    epochs       : int
    imgsz        : int         — input image size (default 640)
    batch        : int
    device_list  : List[int]   — GPU indices (empty = CPU)
    workers      : int
    patience     : int         — early-stopping patience (mAP@0.5)
    log          : LiveLogger
    min_free_gb  : float       — abort if disk free < this value

    Example
    -------
    >>> trainer = YoloTrainer("yolov11x", weights, yaml, out, epochs=100,
    ...                        imgsz=640, batch=16, device_list=[0],
    ...                        workers=4, patience=30, log=log)
    >>> best_pt = trainer.train()
    """

    def __init__(
        self,
        variant:      str,
        weights_path: Path,
        yaml_path:    Path,
        out_dir:      Path,
        epochs:       int,
        imgsz:        int,
        batch:        int,
        device_list:  List[int],
        workers:      int,
        patience:     int,
        log:          LiveLogger,
        min_free_gb:  float = 5.0,
    ):
        self._variant      = variant
        self._weights_path = weights_path
        self._yaml_path    = yaml_path
        self._out_dir      = out_dir
        self._epochs       = epochs
        self._imgsz        = imgsz
        self._batch        = batch
        self._device_list  = device_list
        self._workers      = workers
        self._patience     = patience
        self._log          = log
        self._min_free_gb  = min_free_gb

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """
        Run training and return the path to the best checkpoint.

        Raises
        ------
        ImportError     – ultralytics not installed
        RuntimeError    – disk full, NaN guard triggered, no checkpoint saved
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        self._check_disk()
        self._log_config()

        device_arg = (
            self._device_list[0]
            if len(self._device_list) == 1
            else self._device_list
        )
        model = YOLO(str(self._weights_path))

        # Register callbacks
        model.add_callback(
            "on_train_start",
            make_start_callback(self._variant, self._log),
        )
        model.add_callback(
            "on_train_epoch_end",
            make_epoch_callback(self._variant, self._log, self._epochs),
        )
        model.add_callback(
            "on_train_epoch_end",
            make_nan_guard_callback(self._variant, self._log),
        )
        model.add_callback(
            "on_train_end",
            make_end_callback(self._variant, self._log),
        )

        kw = dict(STABLE_TRAIN_DEFAULTS)
        kw.update(dict(
            data      = str(self._yaml_path),
            epochs    = self._epochs,
            imgsz     = self._imgsz,
            batch     = self._batch,
            device    = device_arg,
            workers   = self._workers,
            patience  = self._patience,
            project   = str(self._out_dir / "yolo_runs"),
            name      = self._variant,
            exist_ok  = True,
        ))
        model.train(**kw)

        return self._find_best_checkpoint()

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _check_disk(self) -> None:
        try:
            free_gb = shutil.disk_usage(str(self._out_dir)).free / (1024 ** 3)
        except Exception:
            free_gb = 999.0
        self._log.log(
            f"[train] Disk free: {free_gb:.1f} GB "
            f"(min {self._min_free_gb:.1f} GB)"
        )
        if free_gb < self._min_free_gb:
            raise RuntimeError(
                f"Disk too full for {self._variant}: {free_gb:.1f} GB"
            )

    def _log_config(self) -> None:
        self._log.sep()
        self._log.log(f"[train] {self._variant}")
        self._log.log(
            f"[train] epochs={self._epochs}  patience={self._patience}  "
            f"batch={self._batch}  imgsz={self._imgsz}  "
            f"workers={self._workers}  device={self._device_list}"
        )
        self._log.log(
            "[train] lr0=3e-4  warmup_bias_lr=0.01  amp=False  (stable config)"
        )

    def _find_best_checkpoint(self) -> Path:
        ckpt_dir = (
            self._out_dir / "yolo_runs" / self._variant / "weights"
        )
        best_pt = ckpt_dir / "best.pt"
        if best_pt.exists():
            self._log.log(f"[train] Best checkpoint: {best_pt}")
            return best_pt

        # Fallback — most recently modified .pt
        pts = sorted(
            ckpt_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not pts:
            raise RuntimeError(
                f"No checkpoint saved for {self._variant}"
            )
        self._log.log(f"[train] Best checkpoint (fallback): {pts[0]}")
        return pts[0]

    def __repr__(self) -> str:
        return (
            f"YoloTrainer("
            f"variant={self._variant!r}, "
            f"epochs={self._epochs}, "
            f"imgsz={self._imgsz}, "
            f"batch={self._batch})"
        )