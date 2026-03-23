# -*- coding: utf-8 -*-
"""
src/training/callbacks.py
==========================
Training callbacks and learning-rate schedulers for ConvNeXt-Base.

Classes
-------
WarmupCosine
    Linear warm-up followed by cosine annealing LR schedule.
EarlyStopping
    Stops training when a monitored metric stops improving.
"""

import numpy as np
from torch.optim import lr_scheduler


# ==============================================================================
# WARMUP + COSINE ANNEALING
# ==============================================================================

class WarmupCosine(lr_scheduler._LRScheduler):
    """
    Linear warm-up for the first ``warmup_epochs`` epochs, then cosine
    annealing from base LR down to ``eta_min``.

    Parameters
    ----------
    optimizer      : torch.optim.Optimizer
    warmup_epochs  : int
    total_epochs   : int
    eta_min        : float   (default 1e-7)
    last_epoch     : int     (default -1)

    Example
    -------
    >>> sched = WarmupCosine(opt, warmup_epochs=5, total_epochs=60)
    >>> for epoch in range(60):
    ...     train(); sched.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        eta_min:       float = 1e-7,
        last_epoch:    int   = -1,
    ):
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup:
            factor = (ep + 1) / max(self.warmup, 1)
            return [base * factor for base in self.base_lrs]
        t   = ep - self.warmup
        T   = max(self.total - self.warmup, 1)
        cos = 0.5 * (1.0 + np.cos(np.pi * t / T))
        return [self.eta_min + (base - self.eta_min) * cos for base in self.base_lrs]

    def __repr__(self) -> str:
        return (
            f"WarmupCosine(warmup={self.warmup}, "
            f"total={self.total}, eta_min={self.eta_min})"
        )


# ==============================================================================
# EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Stops training when ``metric`` has not improved by at least ``min_delta``
    for ``patience`` consecutive epochs.

    Tracks the best value and the best epoch for checkpoint naming.

    Parameters
    ----------
    patience  : int   — consecutive non-improving epochs before stopping
    min_delta : float — minimum improvement to count as progress
    mode      : str   — ``'max'`` (higher = better) or ``'min'``

    Example
    -------
    >>> es = EarlyStopping(patience=12, mode="max")
    >>> for epoch in range(100):
    ...     val_f1 = evaluate()
    ...     if es.step(val_f1):
    ...         print(f"Stopped at epoch {epoch}")
    ...         break
    """

    def __init__(
        self,
        patience:  int   = 12,
        min_delta: float = 1e-4,
        mode:      str   = "max",
    ):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode

        self.best       = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self._counter   = 0

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def step(self, metric: float, epoch: int = 0) -> bool:
        """
        Update the tracker.

        Returns
        -------
        bool — ``True`` if training should stop.
        """
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best       = metric
            self.best_epoch = epoch
            self._counter   = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    @property
    def counter(self) -> int:
        return self._counter

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"mode={self.mode}, best={self.best:.4f})"
        )
