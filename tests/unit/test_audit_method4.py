# -*- coding: utf-8 -*-
"""
tests/unit/test_audit_method4.py
=================================
Unit tests for HardCropProbe.

Tests the transform construction and the interpret helper — no checkpoint,
no GPU, no real dataset required.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from audit.method4_hard_crop import HardCropProbe


# ==============================================================================
# HELPERS
# ==============================================================================

def _probe(tmp_path, image_size=64):
    """Build a HardCropProbe with a non-existent checkpoint (skip mode)."""
    from audit.audit_logger import AuditLogger
    log = AuditLogger(tmp_path / "audit.log")
    return HardCropProbe(
        checkpoint_path="./nonexistent.pth",
        test_dir=tmp_path / "test",
        log=log,
        image_size=image_size,
        batch_size=4,
        num_workers=0,
    ), log


# ==============================================================================
# TESTS
# ==============================================================================

class TestHardCropProbeTransforms:

    def test_standard_tf_output_shape(self, tmp_path):
        probe, log = _probe(tmp_path, image_size=64)
        tf  = probe._make_tf_standard()
        img = Image.new("RGB", (100, 80))
        out = tf(img)
        log.close()
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 64, 64)

    def test_hard_crop_tf_output_shape(self, tmp_path):
        probe, log = _probe(tmp_path, image_size=64)
        tf  = probe._make_tf_hard_crop(crop_size=44)
        img = Image.new("RGB", (100, 100))
        out = tf(img)
        log.close()
        assert out.shape == (3, 64, 64)

    def test_greyscale_tf_3_channels(self, tmp_path):
        """Greyscale transform must still output 3 channels for model compat."""
        probe, log = _probe(tmp_path, image_size=64)
        tf  = probe._make_tf_greyscale()
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        out = tf(img)
        log.close()
        # Shape: 3 channels preserved so model can accept the tensor
        assert out.shape == (3, 64, 64)
        # NOTE: after ImageNet normalisation (per-channel mean/std), channels are
        # NOT numerically equal even from a greyscale source — that is expected.

    def test_combined_tf_output_shape(self, tmp_path):
        probe, log = _probe(tmp_path, image_size=64)
        tf  = probe._make_tf_combined(crop_size=44)
        img = Image.new("RGB", (128, 128))
        out = tf(img)
        log.close()
        assert out.shape == (3, 64, 64)


class TestHardCropProbeInterpret:

    def test_significant_drop(self):
        result = HardCropProbe._interpret("background", 0.15)
        assert "SIGNIFICANT" in result

    def test_moderate_drop(self):
        result = HardCropProbe._interpret("colour", 0.07)
        assert "MODERATE" in result

    def test_minor_drop(self):
        result = HardCropProbe._interpret("layout", 0.03)
        assert "MINOR" in result

    def test_negligible(self):
        result = HardCropProbe._interpret("bg", 0.005)
        assert "NEGLIGIBLE" in result or "robust" in result.lower()

    def test_improvement(self):
        result = HardCropProbe._interpret("bg", -0.05)
        assert "IMPROVEMENT" in result


class TestHardCropProbeSkip:

    def test_skips_when_no_checkpoint(self, tmp_path):
        probe, log = _probe(tmp_path)
        result = probe.run()
        log.close()
        assert result["method"]  == "hard_crop_probe"
        assert result["status"]  == "skipped_no_checkpoint"

    def test_repr(self, tmp_path):
        probe, log = _probe(tmp_path)
        log.close()
        assert "HardCropProbe" in repr(probe)