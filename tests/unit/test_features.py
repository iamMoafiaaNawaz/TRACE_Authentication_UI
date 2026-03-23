# -*- coding: utf-8 -*-
"""
tests/unit/test_features.py
============================
Unit tests for src/features/ — LbpExtractor, FeatureFusion.
"""

import numpy as np
import pytest
from PIL import Image


def _make_grey_image(w: int = 64, h: int = 64) -> np.ndarray:
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


def _make_rgb_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ==============================================================================
# LbpExtractor (if available)
# ==============================================================================

class TestLbpExtractor:

    def test_import(self):
        try:
            from src.features.lbp_extractor import LbpExtractor
        except ImportError as e:
            pytest.skip(f"LbpExtractor not importable: {e}")

    def test_returns_array(self):
        try:
            from src.features.lbp_extractor import LbpExtractor
        except ImportError:
            pytest.skip("LbpExtractor not available")
        extractor = LbpExtractor()
        img = _make_grey_image()
        result = extractor.extract(img)
        assert isinstance(result, np.ndarray)

    def test_output_1d(self):
        try:
            from src.features.lbp_extractor import LbpExtractor
        except ImportError:
            pytest.skip("LbpExtractor not available")
        extractor = LbpExtractor()
        img = _make_grey_image()
        result = extractor.extract(img)
        assert result.ndim == 1

    def test_consistent_length(self):
        try:
            from src.features.lbp_extractor import LbpExtractor
        except ImportError:
            pytest.skip("LbpExtractor not available")
        extractor = LbpExtractor()
        a = extractor.extract(_make_grey_image())
        b = extractor.extract(_make_grey_image())
        assert len(a) == len(b)


# ==============================================================================
# FeatureFusion (if available)
# ==============================================================================

class TestFeatureFusion:

    def test_import(self):
        try:
            from src.features.fusion import FeatureFusion
        except ImportError as e:
            pytest.skip(f"FeatureFusion not importable: {e}")

    def test_concatenation(self):
        try:
            from src.features.fusion import FeatureFusion
        except ImportError:
            pytest.skip("FeatureFusion not available")
        ff = FeatureFusion()
        a  = np.random.rand(128)
        b  = np.random.rand(64)
        result = ff.fuse(a, b)
        assert isinstance(result, np.ndarray)
        assert len(result) >= max(len(a), len(b))
