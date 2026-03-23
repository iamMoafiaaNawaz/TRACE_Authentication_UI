# -*- coding: utf-8 -*-
"""
tests/unit/test_models.py
==========================
Unit tests for src/models/classifier.py

Tests: ConvNeXtClassifier, unwrap_model.
No GPU required — CPU-only checks for architecture and parameter logic.
"""

import pytest
import torch
import torch.nn as nn

from src.models.classifier import ConvNeXtClassifier, unwrap_model


# ==============================================================================
# unwrap_model
# ==============================================================================

class TestUnwrapModel:

    def test_plain_model_returned_unchanged(self):
        model = nn.Linear(10, 5)
        assert unwrap_model(model) is model

    def test_data_parallel_unwrapped(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for DataParallel")
        model = nn.Linear(10, 5).cuda()
        dp    = nn.DataParallel(model)
        assert unwrap_model(dp) is model

    def test_returns_nn_module(self):
        model = nn.Sequential(nn.Linear(4, 4))
        result = unwrap_model(model)
        assert isinstance(result, nn.Module)


# ==============================================================================
# ConvNeXtClassifier
# ==============================================================================

class TestConvNeXtClassifier:

    def test_repr(self):
        clf = ConvNeXtClassifier(num_classes=4, dropout_p=0.4)
        r   = repr(clf)
        assert "ConvNeXtClassifier" in r
        assert "4" in r

    def test_build_returns_nn_module(self):
        clf   = ConvNeXtClassifier(num_classes=4)
        model = clf.build()
        assert isinstance(model, nn.Module)

    def test_output_shape_4_classes(self):
        clf   = ConvNeXtClassifier(num_classes=4, dropout_p=0.0)
        model = clf.build().eval()
        x     = torch.zeros(2, 3, 64, 64)   # small input, no GPU needed
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4)

    def test_output_shape_custom_classes(self):
        clf   = ConvNeXtClassifier(num_classes=7, dropout_p=0.0)
        model = clf.build().eval()
        x     = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 7)

    def test_classifier_head_replaced(self):
        clf   = ConvNeXtClassifier(num_classes=4, dropout_p=0.3)
        model = clf.build()
        head  = unwrap_model(model).classifier[2]
        assert isinstance(head, nn.Sequential)
        # Check GELU is present
        has_gelu = any(isinstance(m, nn.GELU) for m in head.modules())
        assert has_gelu

    def test_set_backbone_grad_frozen(self):
        clf   = ConvNeXtClassifier(num_classes=4)
        model = clf.build()
        ConvNeXtClassifier.set_backbone_grad(model, requires_grad=False)
        m = unwrap_model(model)
        for name, param in m.named_parameters():
            if "classifier" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_set_backbone_grad_unfrozen(self):
        clf   = ConvNeXtClassifier(num_classes=4)
        model = clf.build()
        ConvNeXtClassifier.set_backbone_grad(model, requires_grad=False)
        ConvNeXtClassifier.set_backbone_grad(model, requires_grad=True)
        m = unwrap_model(model)
        for name, param in m.named_parameters():
            if "classifier" not in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_param_groups_length(self):
        clf    = ConvNeXtClassifier(num_classes=4)
        model  = clf.build()
        groups = ConvNeXtClassifier.param_groups(model, lr_head=2e-4, lr_s7=2e-5, lr_rest=5e-6)
        assert len(groups) == 3

    def test_param_groups_lrs(self):
        clf    = ConvNeXtClassifier(num_classes=4)
        model  = clf.build()
        groups = ConvNeXtClassifier.param_groups(model, lr_head=2e-4, lr_s7=2e-5, lr_rest=5e-6)
        assert groups[0]["lr"] == pytest.approx(2e-4)
        assert groups[1]["lr"] == pytest.approx(2e-5)
        assert groups[2]["lr"] == pytest.approx(5e-6)

    def test_param_groups_no_overlap(self):
        clf    = ConvNeXtClassifier(num_classes=4)
        model  = clf.build()
        groups = ConvNeXtClassifier.param_groups(model, lr_head=2e-4, lr_s7=2e-5, lr_rest=5e-6)
        all_ids = []
        for g in groups:
            all_ids.extend([id(p) for p in g["params"]])
        # No parameter should appear in two groups
        assert len(all_ids) == len(set(all_ids))

    def test_all_params_covered(self):
        clf    = ConvNeXtClassifier(num_classes=4)
        model  = clf.build()
        groups = ConvNeXtClassifier.param_groups(model, lr_head=2e-4, lr_s7=2e-5, lr_rest=5e-6)
        group_ids = {id(p) for g in groups for p in g["params"]}
        model_ids = {id(p) for p in model.parameters()}
        assert group_ids == model_ids
