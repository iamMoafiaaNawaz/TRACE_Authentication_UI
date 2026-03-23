# -*- coding: utf-8 -*-
"""
tests/unit/test_audit_method1.py
=================================
Unit tests for ExactHashChecker (Method 1 — MD5 exact duplicate detection).

No GPU, no real images needed — uses synthetic files written to tmp_path.
"""

from pathlib import Path

import pytest

from audit.audit_logger import AuditLogger
from audit.method1_exact_hash import ExactHashChecker


# ==============================================================================
# HELPERS
# ==============================================================================

def _make_splits(tmp_path: Path, duplicate: bool = False):
    """
    Build a minimal splits dict:
      train / BCC / 001.jpg  002.jpg
      val   / BCC / 003.jpg
      test  / BCC / 004.jpg  [optionally 001.jpg copy = duplicate]
    """
    splits = {}
    for split in ("train", "val", "test"):
        d = tmp_path / split / "BCC"
        d.mkdir(parents=True)

    # Unique content per file
    files = {
        "train": ["file_train_1.jpg", "file_train_2.jpg"],
        "val":   ["file_val_1.jpg"],
        "test":  ["file_test_1.jpg"],
    }
    for split, names in files.items():
        for name in names:
            p = tmp_path / split / "BCC" / name
            p.write_bytes(f"content_{split}_{name}".encode())

    # Inject exact duplicate (same bytes as train file in test)
    if duplicate:
        src  = tmp_path / "train" / "BCC" / "file_train_1.jpg"
        dest = tmp_path / "test"  / "BCC" / "DUPE_file.jpg"
        dest.write_bytes(src.read_bytes())

    return {
        split: {"BCC": list((tmp_path / split / "BCC").iterdir())}
        for split in ("train", "val", "test")
    }


def _logger(tmp_path: Path) -> AuditLogger:
    return AuditLogger(tmp_path / "audit.log")


# ==============================================================================
# TESTS
# ==============================================================================

class TestExactHashChecker:

    def test_no_duplicates_passes(self, tmp_path):
        splits = _make_splits(tmp_path, duplicate=False)
        log    = _logger(tmp_path)
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert result["cross_split_duplicates"] == 0

    def test_detects_exact_duplicate(self, tmp_path):
        splits = _make_splits(tmp_path, duplicate=True)
        log    = _logger(tmp_path)
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert result["cross_split_duplicates"] >= 1

    def test_result_has_required_keys(self, tmp_path):
        splits = _make_splits(tmp_path)
        log    = _logger(tmp_path)
        result = ExactHashChecker(log).run(splits)
        log.close()
        for key in ["method", "cross_split_duplicates", "examples"]:
            assert key in result

    def test_method_label(self, tmp_path):
        splits = _make_splits(tmp_path)
        log    = _logger(tmp_path)
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert result["method"] == "exact_md5"

    def test_examples_populated_on_duplicate(self, tmp_path):
        splits = _make_splits(tmp_path, duplicate=True)
        log    = _logger(tmp_path)
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert len(result["examples"]) >= 1
        ex = result["examples"][0]
        assert "hash" in ex
        assert "locations" in ex

    def test_within_split_duplicates_not_counted(self, tmp_path):
        """Two identical files in the SAME split should not be counted."""
        splits = _make_splits(tmp_path)
        # Add a within-train duplicate
        src  = tmp_path / "train" / "BCC" / "file_train_1.jpg"
        dest = tmp_path / "train" / "BCC" / "intra_dupe.jpg"
        dest.write_bytes(src.read_bytes())
        splits["train"]["BCC"].append(dest)

        log    = _logger(tmp_path)
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert result["cross_split_duplicates"] == 0

    def test_repr(self, tmp_path):
        assert "ExactHashChecker" in repr(ExactHashChecker(_logger(tmp_path)))