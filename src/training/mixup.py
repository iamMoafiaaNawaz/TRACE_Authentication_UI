# -*- coding: utf-8 -*-
"""
src/training/mixup.py
======================
Mixup augmentation for ConvNeXt-Base training.

Classes
-------
MixupAugmentation
    Applies Mixup at the batch level and computes the blended loss.

References
----------
Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class MixupAugmentation:
    """
    Batch-level Mixup augmentation.

    Samples a mixing coefficient λ ~ Beta(α, α) and produces a convex
    combination of two randomly permuted batches.  The loss is computed
    as the same convex combination of the per-label cross-entropy.

    Parameters
    ----------
    alpha : float
        Beta distribution parameter.  ``alpha=0`` disables Mixup
        (λ fixed to 1.0, identity transform).

    Example
    -------
    >>> mixup = MixupAugmentation(alpha=0.3)
    >>> x_mix, ya, yb, lam = mixup.apply(x, y)
    >>> loss = mixup.loss(criterion, pred, ya, yb, lam)
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def apply(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Mix a batch.

        Parameters
        ----------
        x : (N, C, H, W) input tensor
        y : (N,) label tensor

        Returns
        -------
        x_mixed, y_a, y_b, lam
        """
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        x_mixed = lam * x + (1.0 - lam) * x[idx]
        return x_mixed, y, y[idx], lam

    @staticmethod
    def loss(
        criterion: nn.Module,
        pred:      torch.Tensor,
        y_a:       torch.Tensor,
        y_b:       torch.Tensor,
        lam:       float,
    ) -> torch.Tensor:
        """
        Compute the mixed loss:  λ * L(pred, y_a) + (1-λ) * L(pred, y_b).
        """
        return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

    def __repr__(self) -> str:
        return f"MixupAugmentation(alpha={self.alpha})"
