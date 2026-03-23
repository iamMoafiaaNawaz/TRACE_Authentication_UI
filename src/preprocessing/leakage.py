# -*- coding: utf-8 -*-
"""
src/preprocessing/leakage.py
=============================
Inline data leakage checker for the training pipeline.

Compares resolved absolute file paths across train / val / test splits
and hard-fails immediately (sys.exit) if any overlap is found — before
a single GPU cycle is wasted on a leaky run.

Also verifies that ``class_to_idx`` is identical across all splits.

Class
-----
LeakageChecker
    Run :meth:`check` after loading all three ImageFolder datasets.
"""

import sys
from typing import Dict

from torchvision import datasets

from src.utils.io_ops import LiveLogger


class LeakageChecker:
    """
    Hard-fail leakage guard for ImageFolder datasets.

    Checks:
    1. Zero file-path overlap across train / val / test
       (uses resolved absolute paths to catch symlinks and renames)
    2. Identical ``class_to_idx`` mapping across all splits

    Parameters
    ----------
    log : LiveLogger

    Example
    -------
    >>> checker = LeakageChecker(log)
    >>> checker.check(train_ds, val_ds, test_ds)   # raises SystemExit on fail
    """

    def __init__(self, log: LiveLogger):
        self._log = log

    def check(
        self,
        train_ds: datasets.ImageFolder,
        val_ds:   datasets.ImageFolder,
        test_ds:  datasets.ImageFolder,
    ) -> None:
        """
        Verify no file-path overlap and consistent class mappings.

        Calls ``sys.exit(1)`` on any leakage — intentional hard fail.
        """
        self._log.log(
            "[leakage_check] Verifying zero file overlap across splits..."
        )

        train_paths = self._resolved(train_ds)
        val_paths   = self._resolved(val_ds)
        test_paths  = self._resolved(test_ds)

        leakage_found = False

        for a_name, a_paths, b_name, b_paths in [
            ("TRAIN", train_paths, "VAL",  val_paths),
            ("TRAIN", train_paths, "TEST", test_paths),
            ("VAL",   val_paths,   "TEST", test_paths),
        ]:
            overlap = a_paths & b_paths
            if overlap:
                self._log.log(
                    f"[leakage_check] CRITICAL: {len(overlap)} files shared "
                    f"between {a_name} and {b_name}!"
                )
                for p in sorted(overlap)[:5]:
                    self._log.log(f"  {p}")
                leakage_found = True

        if leakage_found:
            self._log.log(
                "[leakage_check] ABORTING — fix your dataset splits before training."
            )
            self._log.close()
            sys.exit(1)

        self._log.log(
            f"[leakage_check] PASSED — train/val/test are fully disjoint "
            f"({len(train_paths)} / {len(val_paths)} / {len(test_paths)} unique files)."
        )

        # Verify class_to_idx consistency
        for ds, name in [(val_ds, "val"), (test_ds, "test")]:
            assert ds.class_to_idx == train_ds.class_to_idx, (
                f"class_to_idx mismatch between train and {name}!\n"
                f"  train={train_ds.class_to_idx}\n"
                f"  {name}={ds.class_to_idx}"
            )
        self._log.log(
            "[leakage_check] class_to_idx consistent across all splits."
        )

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _resolved(ds: datasets.ImageFolder) -> set:
        """Return a set of resolved absolute path strings for a dataset."""
        from pathlib import Path
        return {str(Path(p).resolve()) for p, _ in ds.samples}

    def __repr__(self) -> str:
        return "LeakageChecker()"