# -*- coding: utf-8 -*-
"""
src/training/yolo_callbacks.py
==============================
Ultralytics-compatible training callbacks for YOLO experiments.

Functions return callback callables that are registered via
``model.add_callback(event, fn)`` before ``model.train()``.

Callbacks
---------
make_start_callback     — logs training start banner
make_epoch_callback     — logs per-epoch metrics (loss, P/R/F1/mAP, LR, ETA)
make_nan_guard_callback — detects NaN/Inf loss; stops training before
                          checkpoint corruption (2-consecutive-epoch threshold)
make_end_callback       — logs final validation metrics on training completion
"""

import shutil
import time
from pathlib import Path

import numpy as np

from src.utils.io_ops import LiveLogger


# ==============================================================================
# START CALLBACK
# ==============================================================================

def make_start_callback(variant: str, log: LiveLogger):
    """
    Log a training-start banner.

    Registered on event: ``on_train_start``
    """
    def _cb(trainer) -> None:
        log.sep(True)
        log.log(f"[{variant}] TRAINING START  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.log(
            f"[{variant}] epochs={getattr(trainer, 'epochs', '?')}  "
            f"batch={getattr(trainer, 'batch_size', '?')}  "
            f"imgsz={getattr(trainer, 'imgsz', '?')}"
        )
        log.sep(True)
    return _cb


# ==============================================================================
# EPOCH CALLBACK
# ==============================================================================

def make_epoch_callback(variant: str, log: LiveLogger, total_epochs: int):
    """
    Log per-epoch training and validation metrics.

    Tracks the best mAP@0.5 seen so far and marks new bests with ``***``.

    Registered on event: ``on_train_epoch_end``
    """
    t0   = [time.time()]
    best = [0.0]

    def _cb(trainer) -> None:
        ep  = getattr(trainer, "epoch", 0) + 1
        tot = getattr(trainer, "epochs", total_epochs)

        # --- Loss ---
        li  = getattr(trainer, "loss_items", None)
        tl  = getattr(trainer, "tloss",      None)
        bl = cl = dl = 0.0
        if li is not None:
            arr = (li.cpu().numpy()
                   if hasattr(li, "cpu") else np.asarray(li))
            if len(arr) >= 3:
                bl, cl, dl = float(arr[0]), float(arr[1]), float(arr[2])
            elif len(arr) == 2:
                bl, cl = float(arr[0]), float(arr[1])
        total_loss = float(tl) if tl is not None else bl + cl + dl

        # --- Validation metrics ---
        m   = getattr(trainer, "metrics", {}) or {}
        mp  = float(m.get("metrics/precision(B)", 0.0))
        mr  = float(m.get("metrics/recall(B)",    0.0))
        m50 = float(m.get("metrics/mAP50(B)",     0.0))
        m95 = float(m.get("metrics/mAP50-95(B)",  0.0))
        f1  = 2 * mp * mr / (mp + mr + 1e-8)

        # --- Learning rate ---
        lr = 0.0
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            pg = trainer.optimizer.param_groups
            lr = pg[0].get("lr", 0.0) if pg else 0.0

        # --- ETA ---
        elapsed = time.time() - t0[0]
        eta_s   = elapsed / max(ep, 1) * (tot - ep)
        eta     = "%dh%02dm" % (int(eta_s // 3600), int((eta_s % 3600) // 60))

        star = "  *** NEW BEST ***" if m50 > best[0] else ""
        if m50 > best[0]:
            best[0] = m50

        log.sep()
        log.log(f"[{variant}] Epoch {ep}/{tot}  LR={lr:.2e}  ETA={eta}{star}")
        log.log(f"  LOSS   total={total_loss:.4f}  box={bl:.4f}  "
                f"cls={cl:.4f}  dfl={dl:.4f}")
        log.log(f"  VAL    P={mp:.4f}  R={mr:.4f}  F1={f1:.4f}  "
                f"mAP@0.5={m50:.4f}  mAP@0.5:95={m95:.4f}")
        log.log(f"  BEST   mAP@0.5={best[0]:.4f}")
        log.sep()

    return _cb


# ==============================================================================
# NAN GUARD CALLBACK
# ==============================================================================

def make_nan_guard_callback(variant: str, log: LiveLogger):
    """
    Stop training before checkpoint corruption when NaN/Inf loss persists.

    Logic:
    - Detects NaN/Inf in ``trainer.loss_items`` each epoch.
    - If NaN/Inf appears for **2 consecutive epochs**, saves a safe copy of
      ``best.pt`` as ``best_pre_nan.pt``, then raises ``RuntimeError`` to
      abort training cleanly.
    - A single NaN epoch (streak=1) increments the counter but does not stop.

    Registered on event: ``on_train_epoch_end``

    Background
    ----------
    Root cause of the NaN explosion in Exp8 run 1:
    ``warmup_bias_lr=0.1`` (10× too high) + ``amp=True`` combined with
    noisy Otsu pseudo-labels → ``cls_loss`` spike of 157.5 at epoch 4.
    Fixed in ``STABLE_TRAIN_DEFAULTS`` (``warmup_bias_lr=0.01``, ``amp=False``).
    """
    nan_streak = [0]

    def _cb(trainer) -> None:
        import torch
        li = getattr(trainer, "loss_items", None)
        if li is None:
            return

        arr = li if isinstance(li, torch.Tensor) else torch.tensor(li)
        if torch.isnan(arr).any() or torch.isinf(arr).any():
            nan_streak[0] += 1
            log.log(
                f"[nan_guard] {variant} epoch "
                f"{getattr(trainer, 'epoch', 0) + 1}: "
                f"NaN/Inf in loss! streak={nan_streak[0]}"
            )
            if nan_streak[0] >= 2:
                # Save a safe copy before last.pt is overwritten
                best = Path(trainer.save_dir) / "weights" / "best.pt"
                if best.exists():
                    safe = Path(trainer.save_dir) / "weights" / "best_pre_nan.pt"
                    shutil.copy2(str(best), str(safe))
                    log.log(f"[nan_guard] Saved safe copy: {safe}")
                raise RuntimeError(
                    f"[nan_guard] {variant}: NaN loss 2 consecutive epochs. "
                    "Stopping before checkpoint corruption."
                )
        else:
            nan_streak[0] = 0

    return _cb


# ==============================================================================
# END CALLBACK
# ==============================================================================

def make_end_callback(variant: str, log: LiveLogger):
    """
    Log final validation metrics when training completes.

    Registered on event: ``on_train_end``
    """
    def _cb(trainer) -> None:
        m   = getattr(trainer, "metrics", {}) or {}
        mp  = float(m.get("metrics/precision(B)", 0.0))
        mr  = float(m.get("metrics/recall(B)",    0.0))
        m50 = float(m.get("metrics/mAP50(B)",     0.0))
        m95 = float(m.get("metrics/mAP50-95(B)",  0.0))
        log.sep(True)
        log.log(f"[{variant}] TRAINING COMPLETE")
        log.log(
            f"[{variant}] Final Val  P={mp:.4f}  R={mr:.4f}  "
            f"mAP@0.5={m50:.4f}  mAP@0.5:95={m95:.4f}"
        )
        log.sep(True)
    return _cb