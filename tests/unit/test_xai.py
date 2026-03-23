# -*- coding: utf-8 -*-
"""
tests/unit/test_xai.py
========================
Unit tests for:
  src/xai/gradcam.py      — GradCAMPlusPlus.analyse(), get_gradcam_target_layer
  src/xai/visualization.py — TrainingPlotter (file creation, no GPU required)

Pure logic tests — no real model forward pass for GradCAM (mocked cam array).
TrainingPlotter tested with synthetic history dicts.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.xai.gradcam import GradCAMPlusPlus, get_gradcam_target_layer


# ==============================================================================
# HELPERS
# ==============================================================================

def _synthetic_cam(h: int = 32, w: int = 32, pattern: str = "uniform") -> np.ndarray:
    """Create a synthetic saliency map for testing."""
    if pattern == "uniform":
        return np.full((h, w), 0.5, dtype=np.float32)
    elif pattern == "peak_center":
        cam = np.zeros((h, w), dtype=np.float32)
        cam[h // 2, w // 2] = 1.0
        return cam
    elif pattern == "high_activation":
        cam = np.random.uniform(0.8, 1.0, (h, w)).astype(np.float32)
        return cam
    elif pattern == "low_activation":
        cam = np.random.uniform(0.0, 0.1, (h, w)).astype(np.float32)
        return cam
    return np.zeros((h, w), dtype=np.float32)


def _dummy_gradcam_instance():
    """Create a GradCAMPlusPlus with a trivial layer (no real model needed for analyse())."""
    model = nn.Sequential(nn.Linear(4, 4))
    layer = list(model.children())[0]
    cam   = GradCAMPlusPlus(model, layer)
    # Remove hooks so we can test analyse() without running forward
    cam.remove()
    return cam


# ==============================================================================
# GradCAMPlusPlus.analyse()
# ==============================================================================

class TestGradCAMAnalyse:

    def setup_method(self):
        self.cam_gen = _dummy_gradcam_instance()

    def test_returns_dict(self):
        cam    = _synthetic_cam()
        result = self.cam_gen.analyse(cam, "MEL")
        assert isinstance(result, dict)

    def test_required_keys(self):
        cam    = _synthetic_cam()
        result = self.cam_gen.analyse(cam, "BCC")
        for key in ["high_activation_pct", "mid_activation_pct",
                    "mean_activation", "peak_activation",
                    "primary_region", "xai_summary"]:
            assert key in result, f"Missing key: {key}"

    def test_high_activation_pct_uniform(self):
        cam    = _synthetic_cam(pattern="uniform")   # all 0.5, so high% = 0
        result = self.cam_gen.analyse(cam, "MEL")
        assert result["high_activation_pct"] == pytest.approx(0.0)

    def test_high_activation_pct_full(self):
        cam    = np.ones((32, 32), dtype=np.float32)
        result = self.cam_gen.analyse(cam, "MEL")
        assert result["high_activation_pct"] == pytest.approx(100.0, abs=0.1)

    def test_peak_activation_ones(self):
        cam    = np.ones((32, 32), dtype=np.float32)
        result = self.cam_gen.analyse(cam, "NV")
        assert result["peak_activation"] == pytest.approx(1.0)

    def test_peak_activation_zeros(self):
        cam    = np.zeros((32, 32), dtype=np.float32)
        result = self.cam_gen.analyse(cam, "BKL")
        assert result["peak_activation"] == pytest.approx(0.0)

    def test_region_central(self):
        cam             = np.zeros((32, 32), dtype=np.float32)
        cam[15:17, 15:17] = 1.0   # centre
        result          = self.cam_gen.analyse(cam, "MEL")
        assert result["primary_region"] == "central"

    def test_region_diffuse_when_no_high_act(self):
        cam    = np.zeros((32, 32), dtype=np.float32)
        result = self.cam_gen.analyse(cam, "MEL")
        assert result["primary_region"] == "diffuse"

    def test_region_upper_left(self):
        cam             = np.zeros((32, 32), dtype=np.float32)
        cam[0:3, 0:3]   = 1.0    # top-left
        result          = self.cam_gen.analyse(cam, "BCC")
        assert "upper" in result["primary_region"]
        assert "left"  in result["primary_region"]

    def test_xai_summary_contains_class(self):
        cam    = _synthetic_cam(pattern="high_activation")
        result = self.cam_gen.analyse(cam, "MEL")
        assert "MEL" in result["xai_summary"]

    def test_mean_activation_range(self):
        cam    = _synthetic_cam(pattern="uniform")
        result = self.cam_gen.analyse(cam, "NV")
        assert 0.0 <= result["mean_activation"] <= 1.0


# ==============================================================================
# get_gradcam_target_layer
# ==============================================================================

class TestGetTargetLayer:

    def test_returns_nn_module(self):
        from src.models.classifier import ConvNeXtClassifier
        model = ConvNeXtClassifier(num_classes=4).build()
        layer = get_gradcam_target_layer(model)
        assert isinstance(layer, nn.Module)

    def test_target_is_depthwise_conv(self):
        """The target layer should be a Conv2d (depthwise)."""
        from src.models.classifier import ConvNeXtClassifier
        model = ConvNeXtClassifier(num_classes=4).build()
        layer = get_gradcam_target_layer(model)
        assert isinstance(layer, nn.Conv2d)


# ==============================================================================
# TrainingPlotter (file creation)
# ==============================================================================

class TestTrainingPlotter:

    def _synthetic_history(self, n=5):
        return {
            "train_loss":        [0.9 - i * 0.1 for i in range(n)],
            "val_loss":          [1.0 - i * 0.08 for i in range(n)],
            "train_acc":         [0.5 + i * 0.08 for i in range(n)],
            "val_acc":           [0.45 + i * 0.07 for i in range(n)],
            "train_macro_f1":    [0.4 + i * 0.09 for i in range(n)],
            "val_macro_f1":      [0.38 + i * 0.08 for i in range(n)],
            "train_balanced_acc":[0.48 + i * 0.07 for i in range(n)],
            "val_balanced_acc":  [0.43 + i * 0.06 for i in range(n)],
            "lr_head":           [2e-4 * (0.9 ** i) for i in range(n)],
        }

    def _synthetic_split_metrics(self, n_samples=20, n_classes=4):
        labels = list(range(n_classes)) * (n_samples // n_classes)
        preds  = labels[:]
        probs  = np.eye(n_classes)[labels].tolist()
        return {
            "labels": labels,
            "preds":  preds,
            "probs":  probs,
        }

    def test_saves_training_curve_plots(self, tmp_path):
        from src.utils.io_ops import LiveLogger
        from src.xai.visualization import TrainingPlotter
        log     = LiveLogger(tmp_path / "train.log")
        plotter = TrainingPlotter(tmp_path, log)
        plotter.save_training_curves(self._synthetic_history())
        log.close()
        plots_dir = tmp_path / "plots"
        assert plots_dir.exists()
        pngs = list(plots_dir.glob("*.png"))
        assert len(pngs) >= 5, f"Expected >=5 plots, got {len(pngs)}"

    def test_saves_split_plots(self, tmp_path):
        from src.utils.io_ops import LiveLogger
        from src.xai.visualization import TrainingPlotter
        log     = LiveLogger(tmp_path / "train.log")
        plotter = TrainingPlotter(tmp_path, log)
        split_metrics = {"test": self._synthetic_split_metrics()}
        class_names   = ["BCC", "BKL", "MEL", "NV"]
        plotter.save_split_plots(split_metrics, class_names)
        log.close()
        plots_dir = tmp_path / "plots"
        pngs = list(plots_dir.glob("*.png"))
        assert len(pngs) >= 3

    def test_repr(self, tmp_path):
        from src.utils.io_ops import LiveLogger
        from src.xai.visualization import TrainingPlotter
        log     = LiveLogger(tmp_path / "t.log")
        plotter = TrainingPlotter(tmp_path, log)
        log.close()
        assert "TrainingPlotter" in repr(plotter)
