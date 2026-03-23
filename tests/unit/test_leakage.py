# -*- coding: utf-8 -*-
"""
tests/unit/test_leakage.py
===========================
Unit tests for LeakageChecker.

Uses mock ImageFolder objects so no real dataset or GPU is needed.
Tests both the happy path (clean splits) and the hard-fail path
(overlapping files → SystemExit).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.preprocessing.leakage import LeakageChecker


# ==============================================================================
# HELPERS — mock ImageFolder
# ==============================================================================

def _make_ds(samples, class_to_idx=None):
    """Build a mock dataset with .samples and .class_to_idx."""
    ds = MagicMock()
    ds.samples      = samples          # list of (path_str, label_int)
    ds.class_to_idx = class_to_idx or {"BCC": 0, "BKL": 1, "MEL": 2, "NV": 3}
    return ds


def _logger(tmp_path):
    from src.utils.io_ops import LiveLogger
    return LiveLogger(tmp_path / "train.log")


# ==============================================================================
# TESTS
# ==============================================================================

class TestLeakageChecker:

    def test_clean_splits_pass(self, tmp_path):
        """Disjoint paths should pass silently."""
        train_ds = _make_ds([("/data/train/img1.jpg", 0), ("/data/train/img2.jpg", 1)])
        val_ds   = _make_ds([("/data/val/img3.jpg",   0)])
        test_ds  = _make_ds([("/data/test/img4.jpg",  0)])
        log = _logger(tmp_path)
        # Should not raise
        LeakageChecker(log).check(train_ds, val_ds, test_ds)
        log.close()

    def test_train_val_overlap_exits(self, tmp_path):
        """Shared path between train and val must trigger SystemExit."""
        shared = "/data/shared/img.jpg"
        train_ds = _make_ds([(shared, 0), ("/data/train/other.jpg", 1)])
        val_ds   = _make_ds([(shared, 0)])
        test_ds  = _make_ds([("/data/test/img.jpg", 0)])
        log = _logger(tmp_path)
        with pytest.raises(SystemExit):
            LeakageChecker(log).check(train_ds, val_ds, test_ds)

    def test_train_test_overlap_exits(self, tmp_path):
        shared = "/data/shared/img.jpg"
        train_ds = _make_ds([(shared, 0)])
        val_ds   = _make_ds([("/data/val/other.jpg", 0)])
        test_ds  = _make_ds([(shared, 0)])
        log = _logger(tmp_path)
        with pytest.raises(SystemExit):
            LeakageChecker(log).check(train_ds, val_ds, test_ds)

    def test_val_test_overlap_exits(self, tmp_path):
        shared = "/data/shared/img.jpg"
        train_ds = _make_ds([("/data/train/img.jpg", 0)])
        val_ds   = _make_ds([(shared, 0)])
        test_ds  = _make_ds([(shared, 0)])
        log = _logger(tmp_path)
        with pytest.raises(SystemExit):
            LeakageChecker(log).check(train_ds, val_ds, test_ds)

    def test_class_to_idx_mismatch_raises_assertion(self, tmp_path):
        train_ds = _make_ds(
            [("/data/train/img.jpg", 0)],
            class_to_idx={"BCC": 0, "BKL": 1, "MEL": 2, "NV": 3},
        )
        val_ds = _make_ds(
            [("/data/val/img.jpg", 0)],
            class_to_idx={"BCC": 1, "BKL": 0, "MEL": 2, "NV": 3},  # wrong order
        )
        test_ds = _make_ds([("/data/test/img.jpg", 0)])
        log = _logger(tmp_path)
        with pytest.raises(AssertionError):
            LeakageChecker(log).check(train_ds, val_ds, test_ds)

    def test_class_to_idx_consistent_passes(self, tmp_path):
        idx = {"BCC": 0, "BKL": 1, "MEL": 2, "NV": 3}
        train_ds = _make_ds([("/data/train/img.jpg", 0)], class_to_idx=idx)
        val_ds   = _make_ds([("/data/val/img.jpg",   0)], class_to_idx=idx)
        test_ds  = _make_ds([("/data/test/img.jpg",  0)], class_to_idx=idx)
        log = _logger(tmp_path)
        # Should not raise
        LeakageChecker(log).check(train_ds, val_ds, test_ds)
        log.close()

    def test_repr(self, tmp_path):
        log = _logger(tmp_path)
        assert "LeakageChecker" in repr(LeakageChecker(log))
        log.close()