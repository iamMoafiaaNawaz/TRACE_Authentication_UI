# -*- coding: utf-8 -*-
"""
src/training/train_classifier.py
==================================
ConvNeXt-Base training pipeline for TRACE skin lesion classification.

Classes
-------
ConvNeXtTrainer
    Full training loop with progressive fine-tuning, Mixup, AMP,
    early stopping, and all evaluation/saving logic.
"""

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.classifier import ConvNeXtClassifier, unwrap_model
from src.training.callbacks import EarlyStopping, WarmupCosine
from src.training.mixup import MixupAugmentation
from src.utils.io_ops import LiveLogger


class ConvNeXtTrainer:
    """
    Full training pipeline for ConvNeXt-Base skin lesion classification.

    Features
    --------
    - Progressive fine-tuning (backbone frozen during warmup, then unfrozen
      with differential learning rates)
    - Mixup augmentation
    - AMP (automatic mixed precision) with GradScaler
    - Gradient clipping
    - Label smoothing
    - Early stopping on val macro-F1
    - Saves: checkpoint, state-dict, full model, plots, GradCAM++ overlays,
      confusion matrices, ROC/PR curves, clinical reports, JSON metrics

    Parameters
    ----------
    num_classes    : int
    class_names    : list[str]
    out_dir        : Path
    log            : LiveLogger
    device         : torch.device
    epochs         : int          (default 60)
    warmup_epochs  : int          (default 5)
    batch_size     : int          (default 8)
    lr_head        : float        (default 2e-4)
    lr_stage7      : float        (default 2e-5)
    lr_rest        : float        (default 5e-6)
    weight_decay   : float        (default 1e-4)
    label_smoothing: float        (default 0.1)
    dropout        : float        (default 0.4)
    mixup_alpha    : float        (default 0.3)
    grad_clip      : float        (default 1.0)
    patience       : int          (default 12)
    gradcam_samples: int          (default 40)

    Example
    -------
    >>> trainer = ConvNeXtTrainer(
    ...     num_classes=4, class_names=["BCC","BKL","MEL","NV"],
    ...     out_dir=Path("./convNext"), log=log, device=device
    ... )
    >>> trainer.fit(train_loader, val_loader)
    >>> trainer.evaluate_and_save(val_loader, test_loader)
    """

    # Default checkpoint filename
    CKPT_NAME = "best_convnext_checkpoint.pth"

    def __init__(
        self,
        num_classes:     int,
        class_names:     List[str],
        out_dir:         Path,
        log:             LiveLogger,
        device:          torch.device,
        epochs:          int   = 60,
        warmup_epochs:   int   = 5,
        batch_size:      int   = 8,
        lr_head:         float = 2e-4,
        lr_stage7:       float = 2e-5,
        lr_rest:         float = 5e-6,
        weight_decay:    float = 1e-4,
        label_smoothing: float = 0.1,
        dropout:         float = 0.4,
        mixup_alpha:     float = 0.3,
        grad_clip:       float = 1.0,
        patience:        int   = 12,
        gradcam_samples: int   = 40,
        num_workers:     int   = 4,
    ):
        self.num_classes     = num_classes
        self.class_names     = class_names
        self.out_dir         = Path(out_dir)
        self.log             = log
        self.device          = device
        self.epochs          = epochs
        self.warmup_epochs   = warmup_epochs
        self.batch_size      = batch_size
        self.lr_head         = lr_head
        self.lr_stage7       = lr_stage7
        self.lr_rest         = lr_rest
        self.weight_decay    = weight_decay
        self.label_smoothing = label_smoothing
        self.dropout         = dropout
        self.mixup_alpha     = mixup_alpha
        self.grad_clip       = grad_clip
        self.patience        = patience
        self.gradcam_samples = gradcam_samples
        self.num_workers     = num_workers

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = self.out_dir / self.CKPT_NAME

        # Built during fit()
        self._model:    Optional[nn.Module] = None
        self._history:  Dict = {}

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """
        Train the model.

        Returns
        -------
        history : dict of per-epoch metric lists
        """
        self.log.sep(True)
        self.log.log("CONVNEXT-BASE TRAINING PIPELINE")
        self.log.log(f"  Classes:      {self.class_names}")
        self.log.log(f"  Epochs:       {self.epochs}  (warmup={self.warmup_epochs})")
        self.log.log(f"  Batch size:   {self.batch_size}")
        self.log.log(f"  Device:       {self.device}")
        self.log.sep()

        # Build model
        clf   = ConvNeXtClassifier(self.num_classes, self.dropout)
        model = clf.build()

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            self.log.log(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
            model = nn.DataParallel(model)
        model = model.to(self.device)
        self._model = model

        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        mixup     = MixupAugmentation(alpha=self.mixup_alpha)
        scaler    = torch.amp.GradScaler("cuda")
        es        = EarlyStopping(patience=self.patience, mode="max")

        # --- Phase 1: warmup with frozen backbone ---
        ConvNeXtClassifier.set_backbone_grad(model, requires_grad=False)
        param_groups = ConvNeXtClassifier.param_groups(
            model, self.lr_head, self.lr_stage7, self.lr_rest
        )
        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = WarmupCosine(optimizer, self.warmup_epochs, self.epochs)

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_macro_f1": [], "val_macro_f1": [],
            "train_balanced_acc": [], "val_balanced_acc": [],
            "lr_head": [],
        }
        best_f1    = 0.0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # Unfreeze backbone after warmup
            if epoch == self.warmup_epochs + 1:
                self.log.log(f"\n[train] Epoch {epoch}: unfreezing backbone (diff-LR)")
                ConvNeXtClassifier.set_backbone_grad(model, requires_grad=True)

            t0           = time.time()
            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler, mixup
            )
            val_metrics   = self._eval_epoch(model, val_loader, criterion)
            scheduler.step()

            for key in ("loss", "acc", "macro_f1", "balanced_acc"):
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])
            history["lr_head"].append(optimizer.param_groups[0]["lr"])

            elapsed = time.time() - t0
            self.log.log(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['acc']:.4f}  "
                f"val_F1={val_metrics['macro_f1']:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"[{elapsed:.0f}s]"
            )

            # Best model checkpoint
            if val_metrics["macro_f1"] > best_f1:
                best_f1    = val_metrics["macro_f1"]
                best_state = copy.deepcopy(unwrap_model(model).state_dict())
                torch.save({
                    "epoch":              epoch,
                    "model_state_dict":   best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1":  best_f1,
                    "class_names":        self.class_names,
                    "history":            history,
                    "args": {
                        "dropout": self.dropout,
                        "num_classes": self.num_classes,
                    },
                }, self._ckpt_path)
                self.log.log(f"  [ckpt] New best F1={best_f1:.4f} saved.")

            if es.step(val_metrics["macro_f1"], epoch=epoch):
                self.log.log(
                    f"\n[early_stop] No improvement for {self.patience} epochs. "
                    f"Best epoch={es.best_epoch}  Best F1={es.best:.4f}"
                )
                break

        # Restore best weights
        if best_state:
            unwrap_model(model).load_state_dict(best_state)

        self._history = history
        self.log.sep()
        self.log.log(f"[train] Done.  Best val macro-F1={best_f1:.4f}")
        return history

    def evaluate_and_save(
        self,
        val_loader:  DataLoader,
        test_loader: DataLoader,
    ) -> Dict:
        """
        Run final evaluation on val + test, save plots, GradCAM++ overlays,
        and a full JSON metrics summary.
        """
        from evaluation.evaluate_model import ConvNextEvaluator
        from src.xai.gradcam import GradCAMSaver
        from src.xai.visualization import TrainingPlotter

        model = self._model
        if model is None:
            raise RuntimeError("Call fit() before evaluate_and_save().")

        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        evaluator = ConvNextEvaluator(model, self.device, self.num_classes)

        split_metrics = {
            "val":  evaluator.evaluate(val_loader, criterion),
            "test": evaluator.evaluate(test_loader, criterion),
        }

        # Plots
        plotter = TrainingPlotter(self.out_dir, self.log)
        plotter.save_all(self._history, split_metrics, self.class_names)

        # GradCAM++
        cam_out = self.out_dir / "gradcam"
        saver   = GradCAMSaver(model, self.class_names, cam_out, self.device, self.log)
        gradcam_reports = saver.run(test_loader, max_samples=self.gradcam_samples)

        # JSON summary
        summary_path = self.out_dir / "metrics_summary.json"
        if not summary_path.exists():
            clean = lambda sm: {
                k: v for k, v in sm.items()
                if k not in ("preds", "labels", "probs")
            }
            summary = {
                split: clean(metrics)
                for split, metrics in split_metrics.items()
            }
            summary_path.write_text(json.dumps(summary, indent=2, default=str))
            self.log.log(f"[eval] Metrics summary -> {summary_path}")

        return split_metrics

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler:    torch.amp.GradScaler,
        mixup:     MixupAugmentation,
    ) -> Dict:
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        n_samples  = 0

        for inputs, labels in tqdm(loader, leave=False, desc="  train"):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            x_mix, y_a, y_b, lam = mixup.apply(inputs, labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs = model(x_mix)
                loss    = mixup.loss(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward()
            if self.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs          = inputs.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs
            preds       = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "loss":         total_loss / max(n_samples, 1),
            "acc":          float(accuracy_score(y_true, y_pred)),
            "macro_f1":     float(f1),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        }

    def _eval_epoch(
        self,
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
    ) -> Dict:
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, leave=False, desc="  eval"):
                inputs  = inputs.to(self.device, non_blocking=True)
                labels  = labels.to(self.device, non_blocking=True)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                preds   = torch.argmax(outputs, 1)
                total_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        n      = len(loader.dataset)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "loss":         total_loss / n,
            "acc":          float(accuracy_score(y_true, y_pred)),
            "macro_f1":     float(f1),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        }

    def __repr__(self) -> str:
        return (
            f"ConvNeXtTrainer("
            f"num_classes={self.num_classes}, "
            f"epochs={self.epochs}, "
            f"device={self.device})"
        )
