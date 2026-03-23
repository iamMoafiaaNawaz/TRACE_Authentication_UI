# -*- coding: utf-8 -*-
"""
src/models/classifier.py
=========================
ConvNeXt-Base skin lesion classifier.

Classes
-------
ConvNeXtClassifier
    Builds, configures, and manages parameter groups for the
    ConvNeXt-Base classification head with progressive fine-tuning.

Functions
---------
unwrap_model
    Unwraps DataParallel to access the underlying module.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Base_Weights, convnext_base


# ==============================================================================
# HELPERS
# ==============================================================================

def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping ``nn.DataParallel`` if needed."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ==============================================================================
# CLASSIFIER
# ==============================================================================

class ConvNeXtClassifier:
    """
    Builds a ConvNeXt-Base model with a custom classification head for
    skin lesion classification.

    The head replaces the default ``model.classifier[2]`` linear layer with
    a two-layer MLP that includes GELU activations and dropout for
    regularisation.

    Progressive fine-tuning strategy
    ---------------------------------
    1. Warmup phase  — backbone frozen, only head trains.
    2. Unfreeze      — differential learning rates for head, stage-7, rest.

    Parameters
    ----------
    num_classes : int
    dropout_p   : float — dropout in the head (default 0.4)

    Example
    -------
    >>> clf = ConvNeXtClassifier(num_classes=4, dropout_p=0.4)
    >>> model = clf.build()
    >>> param_groups = clf.param_groups(model, lr_head=2e-4, lr_s7=2e-5, lr_rest=5e-6)
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.4):
        self.num_classes = num_classes
        self.dropout_p   = dropout_p

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def build(self) -> nn.Module:
        """
        Construct a ConvNeXt-Base with ImageNet pre-training and a
        custom two-layer classification head.
        """
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_f  = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_f, in_f // 2),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p * 0.5),
            nn.Linear(in_f // 2, self.num_classes),
        )
        return model

    @staticmethod
    def set_backbone_grad(model: nn.Module, requires_grad: bool) -> None:
        """Freeze or unfreeze all backbone layers (everything except the head)."""
        m = unwrap_model(model)
        for name, param in m.named_parameters():
            if "classifier" not in name:
                param.requires_grad = requires_grad

    @staticmethod
    def param_groups(
        model:    nn.Module,
        lr_head:  float,
        lr_s7:    float,
        lr_rest:  float,
    ) -> List[Dict]:
        """
        Build differential LR parameter groups for AdamW.

        Groups
        ------
        - head (classifier)  -> lr_head  (highest LR — training from scratch)
        - stage-7 features   -> lr_s7    (medium LR — fine-tuning last stage)
        - rest of backbone   -> lr_rest  (low LR — slow backbone update)
        """
        m          = unwrap_model(model)
        head_ids   = {id(p) for p in m.classifier.parameters()}
        stage7_ids = {id(p) for p in m.features[7].parameters()}
        return [
            {"params": list(m.classifier.parameters()),  "lr": lr_head},
            {"params": list(m.features[7].parameters()), "lr": lr_s7},
            {"params": [
                p for p in m.parameters()
                if id(p) not in head_ids and id(p) not in stage7_ids
            ], "lr": lr_rest},
        ]

    @staticmethod
    def load_checkpoint(path, device: torch.device):
        """
        Load a training checkpoint.

        Returns
        -------
        (model, class_names, checkpoint_dict)
        """
        from src.models.classifier import ConvNeXtClassifier
        ck          = torch.load(path, map_location=device, weights_only=False)
        class_names = ck.get("class_names", None)
        num_classes = len(class_names) if class_names else 4
        dropout_p   = (ck.get("args", {}) or {}).get("dropout", 0.4)
        clf         = ConvNeXtClassifier(num_classes=num_classes, dropout_p=dropout_p)
        model       = clf.build()
        sd          = ck.get("model_state_dict", ck)
        sd          = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval(), class_names, ck

    def __repr__(self) -> str:
        return (
            f"ConvNeXtClassifier("
            f"num_classes={self.num_classes}, "
            f"dropout_p={self.dropout_p})"
        )
