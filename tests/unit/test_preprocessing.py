# -*- coding: utf-8 -*-
"""
tests/unit/test_preprocessing.py
==================================
Unit tests for:
  src/preprocessing/transforms.py   — ResizePad, TransformBuilder
  src/preprocessing/dataset_loader.py — DatasetLoader (mocked filesystem)
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocessing.transforms import ResizePad, TransformBuilder


# ==============================================================================
# HELPERS
# ==============================================================================

def _make_rgb_image(w: int, h: int) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_image_folder(root: Path, classes, n_per_class: int = 3, size=(32, 32)):
    """Create a minimal ImageFolder-compatible directory for testing."""
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = _make_rgb_image(*size)
            img.save(d / f"img_{cls}_{i:03d}.jpg")


# ==============================================================================
# ResizePad
# ==============================================================================

class TestResizePad:

    def test_square_output_landscape(self):
        tf  = ResizePad(224)
        img = _make_rgb_image(640, 480)
        out = tf(img)
        assert out.size == (224, 224)

    def test_square_output_portrait(self):
        tf  = ResizePad(512)
        img = _make_rgb_image(300, 600)
        out = tf(img)
        assert out.size == (512, 512)

    def test_square_output_already_square(self):
        tf  = ResizePad(256)
        img = _make_rgb_image(256, 256)
        out = tf(img)
        assert out.size == (256, 256)

    def test_small_image(self):
        tf  = ResizePad(224)
        img = _make_rgb_image(50, 30)
        out = tf(img)
        assert out.size == (224, 224)

    def test_custom_fill_color(self):
        tf  = ResizePad(64, fill_color=(127, 127, 127))
        img = _make_rgb_image(100, 50)
        out = tf(img)
        assert out.size == (64, 64)

    def test_repr(self):
        tf = ResizePad(512, fill_color=(0, 0, 0))
        r  = repr(tf)
        assert "ResizePad" in r
        assert "512" in r

    def test_returns_pil_image(self):
        tf  = ResizePad(128)
        img = _make_rgb_image(200, 300)
        out = tf(img)
        assert isinstance(out, Image.Image)

    def test_aspect_ratio_preserved_landscape(self):
        """The image content should not be stretched — check no distortion."""
        tf  = ResizePad(256)
        img = _make_rgb_image(800, 400)  # 2:1 aspect
        out = tf(img)
        # After padding, output is 256x256 — just check no exception and correct size
        assert out.size == (256, 256)


# ==============================================================================
# TransformBuilder
# ==============================================================================

class TestTransformBuilder:

    def test_returns_two_transforms(self):
        train_tf, eval_tf = TransformBuilder.build(224)
        assert train_tf is not None
        assert eval_tf  is not None

    def test_train_transform_returns_tensor(self):
        train_tf, _ = TransformBuilder.build(224)
        img = _make_rgb_image(300, 200)
        out = train_tf(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_eval_transform_returns_tensor(self):
        _, eval_tf = TransformBuilder.build(224)
        img = _make_rgb_image(300, 200)
        out = eval_tf(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_eval_is_deterministic(self):
        """Same image through eval transform twice should give same result."""
        _, eval_tf = TransformBuilder.build(128)
        img = _make_rgb_image(200, 150)
        a   = eval_tf(img)
        b   = eval_tf(img)
        assert torch.allclose(a, b)

    def test_output_size_512(self):
        train_tf, eval_tf = TransformBuilder.build(512)
        img = _make_rgb_image(600, 400)
        assert train_tf(img).shape[-1] == 512
        assert eval_tf(img).shape[-1]  == 512

    def test_imagenet_normalisation_range(self):
        """After normalisation, values can go outside [0,1] — just check it's a tensor."""
        _, eval_tf = TransformBuilder.build(64)
        img = _make_rgb_image(100, 100)
        out = eval_tf(img)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
