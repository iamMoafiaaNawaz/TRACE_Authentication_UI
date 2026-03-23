# -*- coding: utf-8 -*-
"""
tests/integration/test_audit_pipeline.py
==========================================
Integration tests for the full data integrity audit pipeline.

Uses a synthetic image dataset written to tmp_path.
Methods 2 (pHash) and 3 (Embedding) are skipped in CI unless the
optional libraries are available — tested via graceful skip paths.

Run with:
    pytest tests/integration/test_audit_pipeline.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from audit.audit_logger       import AuditLogger
from audit.method1_exact_hash import ExactHashChecker
from audit.method2_phash      import PHashChecker
from audit.method4_hard_crop  import HardCropProbe
from audit.audit_runner       import AuditRunner, collect_image_paths


# ==============================================================================
# SYNTHETIC DATASET FIXTURE
# ==============================================================================

NUM_CLASSES = 4
CLASSES     = ["BCC", "BKL", "MEL", "NV"]
IMG_SIZE    = 32


def _write_image(path: Path, color=(128, 64, 32)):
    arr = np.full((IMG_SIZE, IMG_SIZE, 3), color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_dataset(root: Path, n_per_class: int = 4, inject_dupe: bool = False):
    """Create a synthetic ImageFolder dataset across train/validation/test.

    Colors are unique per (split, class, index) so that no two images are
    byte-identical — avoiding false positives in the clean-dataset tests.
    """
    splits = ("train", "validation", "test")
    for s_idx, split in enumerate(splits):
        for cls in CLASSES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            cls_idx = CLASSES.index(cls)
            for i in range(n_per_class):
                # Unique color: use split index to avoid cross-split collisions
                r = (s_idx * 80 + cls_idx * 20 + i * 5) % 256
                g = (s_idx * 50 + cls_idx * 30 + i * 7 + 10) % 256
                b = (s_idx * 30 + cls_idx * 10 + i * 11 + 20) % 256
                _write_image(d / f"{i:03d}.jpg", color=(r, g, b))

    if inject_dupe:
        src  = root / "train" / "BCC" / "000.jpg"
        dest = root / "test"  / "BCC" / "DUPE_from_train.jpg"
        dest.write_bytes(src.read_bytes())


@pytest.fixture(scope="module")
def clean_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean_ds")
    _make_dataset(root, n_per_class=4, inject_dupe=False)
    return root


@pytest.fixture(scope="module")
def leaky_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("leaky_ds")
    _make_dataset(root, n_per_class=4, inject_dupe=True)
    return root


def _splits(root: Path):
    val_dir = root / "validation" if (root / "validation").exists() else root / "val"
    return {
        "train":      collect_image_paths(root / "train"),
        "validation": collect_image_paths(val_dir),
        "test":       collect_image_paths(root / "test"),
    }


# ==============================================================================
# collect_image_paths
# ==============================================================================

class TestCollectImagePaths:

    def test_returns_all_classes(self, clean_dataset):
        result = collect_image_paths(clean_dataset / "train")
        assert set(result.keys()) == set(CLASSES)

    def test_correct_image_count(self, clean_dataset):
        result = collect_image_paths(clean_dataset / "train")
        for cls in CLASSES:
            assert len(result[cls]) == 4

    def test_paths_are_path_objects(self, clean_dataset):
        result = collect_image_paths(clean_dataset / "train")
        for paths in result.values():
            for p in paths:
                assert isinstance(p, Path)


# ==============================================================================
# Method 1 — exact hash (integration)
# ==============================================================================

class TestMethod1Integration:

    def test_clean_dataset_passes(self, clean_dataset, tmp_path):
        splits = _splits(clean_dataset)
        log    = AuditLogger(tmp_path / "audit.log")
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert result["cross_split_duplicates"] == 0

    def test_leaky_dataset_detected(self, leaky_dataset, tmp_path):
        splits = _splits(leaky_dataset)
        log    = AuditLogger(tmp_path / "audit.log")
        result = ExactHashChecker(log).run(splits)
        log.close()
        assert result["cross_split_duplicates"] >= 1
        assert len(result["examples"]) >= 1

    def test_log_file_written(self, clean_dataset, tmp_path):
        splits = _splits(clean_dataset)
        log    = AuditLogger(tmp_path / "audit.log")
        ExactHashChecker(log).run(splits)
        log.close()
        assert (tmp_path / "audit.log").exists()
        assert len((tmp_path / "audit.log").read_text()) > 100


# ==============================================================================
# Method 2 — pHash (integration, skips if imagededup missing)
# ==============================================================================

class TestMethod2Integration:

    def test_clean_dataset_passes_or_skips(self, clean_dataset, tmp_path):
        splits = _splits(clean_dataset)
        log    = AuditLogger(tmp_path / "audit.log")
        result = PHashChecker(log, max_hamming_distance=10).run(splits)
        log.close()
        # Either passes with 0 cross-split clusters OR skips gracefully
        if result.get("status") == "skipped_missing_imagededup":
            pytest.skip("imagededup not installed")
        assert result["cross_split_clusters"] == 0

    def test_graceful_skip_without_imagededup(self, clean_dataset, tmp_path, monkeypatch):
        """Force imagededup import to fail and verify graceful skip."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "imagededup" or name.startswith("imagededup"):
                raise ImportError("mocked missing imagededup")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        splits = _splits(clean_dataset)
        log    = AuditLogger(tmp_path / "audit2.log")
        result = PHashChecker(log).run(splits)
        log.close()
        assert result.get("status") == "skipped_missing_imagededup"


# ==============================================================================
# Method 4 — Hard Crop Probe (no checkpoint = skips)
# ==============================================================================

class TestMethod4Integration:

    def test_skips_without_checkpoint(self, clean_dataset, tmp_path):
        log    = AuditLogger(tmp_path / "audit.log")
        probe  = HardCropProbe(
            checkpoint_path="./nonexistent.pth",
            test_dir=clean_dataset / "test",
            log=log, image_size=32, batch_size=4, num_workers=0,
        )
        result = probe.run()
        log.close()
        assert result["status"] == "skipped_no_checkpoint"


# ==============================================================================
# AuditRunner — end-to-end (skips embedding + hard crop, fast)
# ==============================================================================

class TestAuditRunnerIntegration:

    def test_runner_produces_report_json(self, clean_dataset, tmp_path):
        runner = AuditRunner(
            data_root          = str(clean_dataset),
            audit_out          = tmp_path / "audit_out",
            checkpoint_path    = None,
            image_size         = 32,
            batch_size         = 4,
            num_workers        = 0,
            skip_phash         = True,   # skip optional dep
            skip_embedding     = True,   # skip GPU-heavy step
            skip_hard_crop     = True,   # skip checkpoint-dependent step
        )
        results = runner.run()

        # JSON report written
        report_path = tmp_path / "audit_out" / "audit_report.json"
        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert "method1_exact_hash" in loaded

    def test_runner_clean_data_passes_method1(self, clean_dataset, tmp_path):
        runner = AuditRunner(
            data_root       = str(clean_dataset),
            audit_out       = tmp_path / "audit_out2",
            skip_phash      = True,
            skip_embedding  = True,
            skip_hard_crop  = True,
        )
        results = runner.run()
        m1 = results.get("method1_exact_hash", {})
        assert m1.get("cross_split_duplicates", 0) == 0

    def test_runner_leaky_data_detected(self, leaky_dataset, tmp_path):
        runner = AuditRunner(
            data_root       = str(leaky_dataset),
            audit_out       = tmp_path / "audit_out3",
            skip_phash      = True,
            skip_embedding  = True,
            skip_hard_crop  = True,
        )
        results = runner.run()
        m1 = results.get("method1_exact_hash", {})
        assert m1.get("cross_split_duplicates", 0) >= 1

    def test_runner_log_file_created(self, clean_dataset, tmp_path):
        runner = AuditRunner(
            data_root       = str(clean_dataset),
            audit_out       = tmp_path / "audit_out4",
            skip_phash      = True,
            skip_embedding  = True,
            skip_hard_crop  = True,
        )
        runner.run()
        assert (tmp_path / "audit_out4" / "audit_log.txt").exists()