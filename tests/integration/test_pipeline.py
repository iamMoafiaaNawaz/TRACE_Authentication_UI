# -*- coding: utf-8 -*-
"""
tests/integration/test_pipeline.py
=====================================
Integration tests for the ConvNeXt classification pipeline.

Tests the key end-to-end data flows:
  1. DatasetLoader -> DataLoader creation
  2. ConvNeXtClassifier.build() -> forward pass -> loss
  3. MixupAugmentation applied during training step
  4. WarmupCosine LR schedule
  5. EarlyStopping

Uses tiny synthetic datasets — no real images or checkpoints needed.
Does NOT run the full training loop (too slow for CI).
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.models.classifier     import ConvNeXtClassifier
from src.training.callbacks    import EarlyStopping, WarmupCosine
from src.training.mixup        import MixupAugmentation
from src.preprocessing.transforms import TransformBuilder


# ==============================================================================
# HELPERS
# ==============================================================================

CLASS_NAMES = ["BCC", "BKL", "MEL", "NV"]


def _make_image_folder(root: Path, classes=CLASS_NAMES, n=4, size=(32, 32)):
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{i:03d}.jpg")


def _make_split_dataset(tmp_path):
    """Create a minimal train/validation/test split."""
    for split in ("train", "validation", "test"):
        _make_image_folder(tmp_path / split, n=4)
    return tmp_path


# ==============================================================================
# Dataset Loading Integration
# ==============================================================================

class TestDatasetLoaderIntegration:

    def test_load_returns_three_datasets(self, tmp_path):
        from src.utils.io_ops import LiveLogger
        from src.preprocessing.dataset_loader import DatasetLoader
        root = _make_split_dataset(tmp_path / "data")
        log  = LiveLogger(tmp_path / "t.log")
        loader = DatasetLoader(root, image_size=32, log=log)
        train_ds, val_ds, test_ds = loader.load()
        log.close()
        assert len(train_ds) > 0
        assert len(val_ds)   > 0
        assert len(test_ds)  > 0

    def test_class_names_consistent(self, tmp_path):
        from src.utils.io_ops import LiveLogger
        from src.preprocessing.dataset_loader import DatasetLoader
        root = _make_split_dataset(tmp_path / "data")
        log  = LiveLogger(tmp_path / "t.log")
        loader = DatasetLoader(root, image_size=32, log=log)
        train_ds, val_ds, test_ds = loader.load()
        log.close()
        assert train_ds.classes == val_ds.classes == test_ds.classes

    def test_build_loaders_returns_dataloaders(self, tmp_path):
        from torch.utils.data import DataLoader
        from src.utils.io_ops import LiveLogger
        from src.preprocessing.dataset_loader import DatasetLoader
        root = _make_split_dataset(tmp_path / "data")
        log  = LiveLogger(tmp_path / "t.log")
        loader = DatasetLoader(root, image_size=32, log=log)
        train_loader, val_loader, test_loader = loader.build_loaders(
            batch_size=2, num_workers=0
        )
        log.close()
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader,   DataLoader)
        assert isinstance(test_loader,  DataLoader)


# ==============================================================================
# Model Forward Pass Integration
# ==============================================================================

class TestModelForwardPass:

    def test_forward_pass_cpu(self):
        clf   = ConvNeXtClassifier(num_classes=4, dropout_p=0.0)
        model = clf.build().eval()
        x     = torch.zeros(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)
        assert not torch.isnan(out).any()

    def test_loss_finite(self):
        clf       = ConvNeXtClassifier(num_classes=4, dropout_p=0.0)
        model     = clf.build().eval()
        criterion = nn.CrossEntropyLoss()
        x         = torch.zeros(4, 3, 64, 64)
        y         = torch.randint(0, 4, (4,))
        with torch.no_grad():
            out  = model(x)
            loss = criterion(out, y)
        assert torch.isfinite(loss)


# ==============================================================================
# Mixup Integration
# ==============================================================================

class TestMixupIntegration:

    def test_mixup_output_shape_preserved(self):
        mixup = MixupAugmentation(alpha=0.3)
        x     = torch.rand(8, 3, 32, 32)
        y     = torch.randint(0, 4, (8,))
        x_mix, y_a, y_b, lam = mixup.apply(x, y)
        assert x_mix.shape == x.shape
        assert y_a.shape   == y.shape
        assert y_b.shape   == y.shape

    def test_mixup_loss_finite(self):
        mixup     = MixupAugmentation(alpha=0.3)
        criterion = nn.CrossEntropyLoss()
        clf       = ConvNeXtClassifier(num_classes=4, dropout_p=0.0)
        model     = clf.build().eval()

        x = torch.zeros(4, 3, 64, 64)
        y = torch.randint(0, 4, (4,))
        x_mix, y_a, y_b, lam = mixup.apply(x, y)

        with torch.no_grad():
            pred = model(x_mix)
            loss = MixupAugmentation.loss(criterion, pred, y_a, y_b, lam)
        assert torch.isfinite(loss)

    def test_alpha_zero_is_identity(self):
        mixup = MixupAugmentation(alpha=0.0)
        x     = torch.rand(4, 3, 32, 32)
        y     = torch.randint(0, 4, (4,))
        x_mix, y_a, y_b, lam = mixup.apply(x, y)
        assert lam == pytest.approx(1.0)
        assert torch.allclose(x_mix, x)


# ==============================================================================
# WarmupCosine LR Scheduler
# ==============================================================================

class TestWarmupCosineIntegration:

    def _simple_optimizer(self, lr=1e-3):
        model = nn.Linear(10, 4)
        return torch.optim.AdamW(model.parameters(), lr=lr)

    def test_warmup_increases_lr(self):
        opt   = self._simple_optimizer(lr=1e-4)
        sched = WarmupCosine(opt, warmup_epochs=5, total_epochs=20)
        lrs   = []
        for _ in range(5):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # LR should generally increase during warmup
        assert lrs[-1] >= lrs[0]

    def test_cosine_decreases_after_warmup(self):
        opt   = self._simple_optimizer(lr=1e-3)
        sched = WarmupCosine(opt, warmup_epochs=2, total_epochs=10)
        for _ in range(3):  # past warmup
            sched.step()
        lr_after_warmup = opt.param_groups[0]["lr"]
        for _ in range(5):
            sched.step()
        lr_late = opt.param_groups[0]["lr"]
        assert lr_late <= lr_after_warmup

    def test_reaches_eta_min(self):
        eta_min = 1e-7
        opt     = self._simple_optimizer(lr=1e-3)
        sched   = WarmupCosine(opt, warmup_epochs=1, total_epochs=5, eta_min=eta_min)
        for _ in range(6):
            sched.step()
        assert opt.param_groups[0]["lr"] >= eta_min - 1e-12


# ==============================================================================
# EarlyStopping
# ==============================================================================

class TestEarlyStoppingIntegration:

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode="max")
        # First call improves
        assert not es.step(0.5, epoch=1)
        # Next 3 calls don't improve
        assert not es.step(0.4, epoch=2)
        assert not es.step(0.4, epoch=3)
        assert es.step(0.4, epoch=4)   # should stop

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.5, epoch=1)
        es.step(0.4, epoch=2)  # counter=1
        es.step(0.6, epoch=3)  # improvement — counter resets
        assert es.counter == 0

    def test_best_epoch_tracked(self):
        es = EarlyStopping(patience=5, mode="max")
        es.step(0.5, epoch=1)
        es.step(0.8, epoch=3)
        es.step(0.7, epoch=5)
        assert es.best_epoch == 3

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min")
        assert not es.step(1.0, epoch=1)  # first, improves
        assert not es.step(0.9, epoch=2)  # improves
        assert not es.step(0.95, epoch=3) # no improve, counter=1
        assert es.step(0.95, epoch=4)     # no improve, counter=2, stop
