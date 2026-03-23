# -*- coding: utf-8 -*-
"""
src/utils/seed_all.py
=====================
Reproducibility utility — seeds all RNGs used in the TRACE pipeline.
"""

import os
import random

import numpy as np
import torch


class Seeder:
    """
    Centralised seeding for full experiment reproducibility.

    Sets seeds for Python ``random``, NumPy, PyTorch (CPU + all GPUs),
    and configures cuDNN determinism.

    Parameters
    ----------
    seed : int
        Master random seed.

    Example
    -------
    >>> seeder = Seeder(seed=42)
    >>> seeder.seed_everything()
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def seed_everything(self) -> None:
        """Apply seed to all relevant RNG sources."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def __repr__(self) -> str:
        return f"Seeder(seed={self.seed})"