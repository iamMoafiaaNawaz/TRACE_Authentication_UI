# -*- coding: utf-8 -*-
"""
src/preprocessing/dataset_loader.py
=====================================
Loads pre-split ImageFolder datasets and enforces zero data leakage.

Classes
-------
DatasetLoader
    Loads train / val / test splits, asserts class consistency, and
    delegates to LeakageChecker for hard-fail leakage validation.
"""

import sys
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets

from src.preprocessing.leakage import LeakageChecker
from src.preprocessing.transforms import TransformBuilder
from src.utils.io_ops import LiveLogger


class DatasetLoader:
    """
    Loads pre-split skin lesion datasets (train / validation / test).

    Supports both ``validation/`` and ``val/`` folder naming conventions.
    Runs LeakageChecker before returning — training will not start if any
    file overlaps are detected.

    Parameters
    ----------
    data_root  : str or Path
    image_size : int
    log        : LiveLogger

    Example
    -------
    >>> loader = DatasetLoader("./data/split", image_size=512, log=log)
    >>> train_ds, val_ds, test_ds = loader.load()
    """

    def __init__(self, data_root, image_size: int, log: LiveLogger):
        self._root       = Path(data_root)
        self._image_size = image_size
        self._log        = log

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def load(self):
        """
        Returns
        -------
        (train_ds, val_ds, test_ds) : torchvision ImageFolder datasets
        """
        train_tf, eval_tf = TransformBuilder.build(self._image_size)
        val_dir = self._resolve_val_dir()

        self._assert_dirs_exist(val_dir)

        train_ds = datasets.ImageFolder(str(self._root / "train"),  transform=train_tf)
        val_ds   = datasets.ImageFolder(str(val_dir),               transform=eval_tf)
        test_ds  = datasets.ImageFolder(str(self._root / "test"),   transform=eval_tf)

        assert train_ds.classes == val_ds.classes == test_ds.classes, (
            f"Class mismatch! "
            f"train={train_ds.classes} val={val_ds.classes} test={test_ds.classes}"
        )

        self._log.log(f"[data] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
        self._log.log(f"[data] Classes ({len(train_ds.classes)}): {train_ds.classes}")
        self._log.log(f"[data] Val folder: {val_dir}")

        checker = LeakageChecker(log=self._log)
        checker.check(train_ds, val_ds, test_ds)

        return train_ds, val_ds, test_ds

    def build_loaders(
        self,
        batch_size:  int,
        num_workers: int,
        pin_memory:  bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Convenience wrapper: load datasets and immediately build DataLoaders.

        Returns
        -------
        (train_loader, val_loader, test_loader)
        """
        train_ds, val_ds, test_ds = self.load()
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _resolve_val_dir(self) -> Path:
        val = self._root / "validation"
        return val if val.exists() else self._root / "val"

    def _assert_dirs_exist(self, val_dir: Path) -> None:
        for d, name in [
            (self._root / "train", "train"),
            (val_dir, "val/validation"),
            (self._root / "test", "test"),
        ]:
            if not d.exists():
                raise FileNotFoundError(
                    f"'{name}' folder not found: {d}\n"
                    f"Check --data_root path."
                )

    def __repr__(self) -> str:
        return f"DatasetLoader(root={self._root}, image_size={self._image_size})"
