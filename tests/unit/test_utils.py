# -*- coding: utf-8 -*-
"""
tests/test_utils.py
===================
Unit tests for src/utils/ — Seeder, LiveLogger, JsonIO,
WorkerResolver, TrainingConfig, ConfigLoader.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.seed_all      import Seeder
from src.utils.io_ops        import LiveLogger, JsonIO, WorkerResolver
from src.utils.config_loader import TrainingConfig, ConfigLoader


# ==============================================================================
# Seeder
# ==============================================================================

class TestSeeder:

    def test_repr(self):
        assert "Seeder(seed=42)" == repr(Seeder(42))

    def test_seed_everything_runs(self):
        # Should not raise
        Seeder(0).seed_everything()

    def test_reproducibility(self):
        Seeder(42).seed_everything()
        a = torch.rand(5)
        Seeder(42).seed_everything()
        b = torch.rand(5)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        Seeder(1).seed_everything()
        a = torch.rand(5)
        Seeder(2).seed_everything()
        b = torch.rand(5)
        assert not torch.allclose(a, b)

    def test_sets_pythonhashseed(self):
        Seeder(99).seed_everything()
        assert os.environ["PYTHONHASHSEED"] == "99"


# ==============================================================================
# LiveLogger
# ==============================================================================

class TestLiveLogger:

    def test_creates_file(self, tmp_path):
        log = LiveLogger(tmp_path / "train.log")
        log.close()
        assert (tmp_path / "train.log").exists()

    def test_log_writes_content(self, tmp_path):
        log = LiveLogger(tmp_path / "train.log")
        log.log("hello world")
        log.close()
        content = (tmp_path / "train.log").read_text()
        assert "hello world" in content

    def test_sep_light(self, tmp_path):
        log = LiveLogger(tmp_path / "train.log")
        log.sep(heavy=False)
        log.close()
        assert "-" * 80 in (tmp_path / "train.log").read_text()

    def test_sep_heavy(self, tmp_path):
        log = LiveLogger(tmp_path / "train.log")
        log.sep(heavy=True)
        log.close()
        assert "=" * 80 in (tmp_path / "train.log").read_text()

    def test_creates_parent_dirs(self, tmp_path):
        log = LiveLogger(tmp_path / "nested" / "dir" / "train.log")
        log.close()
        assert (tmp_path / "nested" / "dir" / "train.log").exists()

    def test_repr(self, tmp_path):
        log = LiveLogger(tmp_path / "train.log")
        log.close()
        assert "LiveLogger" in repr(log)


# ==============================================================================
# JsonIO
# ==============================================================================

class TestJsonIO:

    def test_save_and_load_dict(self, tmp_path):
        data = {"epoch": 5, "val_f1": 0.87}
        p    = tmp_path / "out.json"
        JsonIO.save(data, p)
        loaded = JsonIO.load(p)
        assert loaded == data

    def test_save_and_load_list(self, tmp_path):
        data = [1, 2, 3, 4]
        p    = tmp_path / "list.json"
        JsonIO.save(data, p)
        assert JsonIO.load(p) == data

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "metrics.json"
        JsonIO.save({"a": 1}, p)
        assert p.exists()

    def test_indent_formatting(self, tmp_path):
        p = tmp_path / "pretty.json"
        JsonIO.save({"key": "value"}, p, indent=4)
        raw = p.read_text()
        assert "\n" in raw   # indented means multi-line


# ==============================================================================
# WorkerResolver
# ==============================================================================

class TestWorkerResolver:

    def test_explicit_override(self):
        assert WorkerResolver.resolve(explicit=7) == 7

    def test_explicit_zero(self):
        assert WorkerResolver.resolve(explicit=0) == 0

    def test_auto_returns_int(self):
        n = WorkerResolver.resolve(explicit=None)
        assert isinstance(n, int)
        assert n >= 0

    def test_auto_within_bounds(self):
        n = WorkerResolver.resolve(explicit=None)
        assert n <= max(4, os.cpu_count() or 4)

    def test_slurm_env(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        n = WorkerResolver.resolve(explicit=None)
        assert n == 6   # max(2, 8-2)
        monkeypatch.delenv("SLURM_CPUS_PER_TASK")


# ==============================================================================
# TrainingConfig
# ==============================================================================

class TestTrainingConfig:

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs        == 60
        assert cfg.warmup_epochs == 5
        assert cfg.patience      == 12
        assert cfg.lr_head       == pytest.approx(2e-4)
        assert cfg.seed          == 42

    def test_to_dict_keys(self):
        cfg = TrainingConfig()
        d   = cfg.to_dict()
        for key in ["epochs", "lr_head", "dropout", "image_size", "seed"]:
            assert key in d

    def test_to_dict_values(self):
        cfg = TrainingConfig(epochs=10, seed=7)
        d   = cfg.to_dict()
        assert d["epochs"] == 10
        assert d["seed"]   == 7

    def test_repr_contains_class_name(self):
        assert "TrainingConfig" in repr(TrainingConfig())

    def test_mutate_field(self):
        cfg = TrainingConfig()
        cfg.epochs = 100
        assert cfg.epochs == 100