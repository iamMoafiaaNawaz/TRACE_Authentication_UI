# This file will evaluate trained models
# -*- coding: utf-8 -*-
"""
evaluation/evaluate_model.py
=============================
Model evaluation entry points for both pipelines.

Classes
-------
ConvNextEvaluator
    Runs inference over a DataLoader, collects predictions/probabilities,
    then delegates all metric computation to ``MetricsCalculator``.
    Used in ``experiments/train_convnext.py`` after every training epoch
    and for final train/val/test reporting.

YoloEvaluator
    Wraps ``ultralytics`` ``.val()`` for a given split and delegates metric
    extraction to ``YoloMetricsExtractor``.
    Used in ``experiments/train_yolo_exp8.py`` after training completes.
"""

import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from evaluation.metrics import MetricsCalculator, YoloMetricsExtractor
from src.utils.io_ops import LiveLogger


# ==============================================================================
# CONVNEXT EVALUATOR
# ==============================================================================

class ConvNextEvaluator:
    """
    Evaluates a ConvNeXt-Base classifier on a single DataLoader split.

    Delegates all metric computation to :class:`MetricsCalculator`.

    Parameters
    ----------
    num_classes : int

    Example
    -------
    >>> evaluator = ConvNextEvaluator(num_classes=4)
    >>> metrics = evaluator.evaluate(model, val_loader, criterion, device)
    >>> print(f"val_macro_f1 = {metrics['macro_f1']:.4f}")
    """

    def __init__(self, num_classes: int):
        self._num_classes = num_classes
        self._calculator  = MetricsCalculator(num_classes)

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model:     nn.Module,
        loader:    torch.utils.data.DataLoader,
        criterion: nn.Module,
        device:    torch.device,
        num_classes: int = None,   # optional override
    ) -> Dict:
        """
        Run inference over ``loader`` and compute all classification metrics.

        Parameters
        ----------
        model     : nn.Module  — trained model (eval mode set internally)
        loader    : DataLoader
        criterion : nn.Module  — loss function (e.g. CrossEntropyLoss)
        device    : torch.device
        num_classes : int or None — overrides constructor value if provided

        Returns
        -------
        dict — see :class:`MetricsCalculator` for full key list.
              Always includes ``preds``, ``labels``, ``probs`` arrays.
        """
        nc = num_classes or self._num_classes
        model.eval()

        total_loss = 0.0
        all_preds:  List[int]   = []
        all_labels: List[int]   = []
        all_probs:  List[list]  = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, leave=False, desc="  eval"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                probs   = torch.softmax(outputs, 1)
                preds   = torch.argmax(outputs, 1)

                total_loss  += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        return self._calculator.compute(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            total_loss=total_loss,
            n_samples=len(loader.dataset),
        )

    def evaluate_all_splits(
        self,
        model:            nn.Module,
        loaders:          Dict[str, torch.utils.data.DataLoader],
        criterion:        nn.Module,
        device:           torch.device,
        log:              LiveLogger,
    ) -> Dict[str, Dict]:
        """
        Evaluate on multiple splits and log a summary line per split.

        Parameters
        ----------
        loaders : dict mapping split name → DataLoader
                  e.g. ``{"train": ..., "validation": ..., "test": ...}``

        Returns
        -------
        dict mapping split name → metrics dict
        """
        results: Dict[str, Dict] = {}
        for split_name, loader in loaders.items():
            metrics = self.evaluate(model, loader, criterion, device)
            results[split_name] = metrics
            log.log(
                f"  {split_name:12s} | "
                f"acc={metrics['acc']:.5f}  "
                f"bal={metrics['balanced_acc']:.5f}  "
                f"F1={metrics['macro_f1']:.5f}  "
                f"MCC={metrics['mcc']:.4f}  "
                f"AUC={metrics.get('macro_auc_ovr') or 0:.4f}  "
                f"PR-AUC={metrics.get('macro_pr_auc') or 0:.4f}"
            )
        return results

    def __repr__(self) -> str:
        return f"ConvNextEvaluator(num_classes={self._num_classes})"


# ==============================================================================
# YOLO EVALUATOR
# ==============================================================================

class YoloEvaluator:
    """
    Evaluates a YOLO model on val and test splits using ultralytics ``.val()``.

    Delegates metric extraction to :class:`YoloMetricsExtractor`.

    Parameters
    ----------
    class_names : List[str]
    log         : LiveLogger

    Example
    -------
    >>> evaluator = YoloEvaluator(class_names=["BCC","BKL","MEL","NV"], log=log)
    >>> results = evaluator.evaluate_all(
    ...     best_pt=Path("best.pt"),
    ...     yaml_path=Path("dataset.yaml"),
    ...     out_dir=Path("./exp8"),
    ...     variant="yolov11x",
    ...     imgsz=640,
    ...     batch=16,
    ...     device_list=[0],
    ... )
    >>> print(results["test"]["mAP_50"])
    """

    def __init__(self, class_names: List[str], log: LiveLogger):
        self._class_names = class_names
        self._log         = log
        self._extractor   = YoloMetricsExtractor(class_names)

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def evaluate_split(
        self,
        best_pt:   Path,
        yaml_path: Path,
        out_dir:   Path,
        variant:   str,
        split:     str,
        imgsz:     int,
        batch:     int,
        device:    int,
    ) -> Dict:
        """
        Run ``model.val()`` on a single split and return the metrics dict.

        Returns ``{"error": <msg>, "mAP_50": 0., "mAP_50_95": 0.}`` on failure.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        self._log.log(f"[eval] {variant} / {split} ...")
        try:
            model  = YOLO(str(best_pt))
            result = model.val(
                data    = str(yaml_path),
                split   = split,
                imgsz   = imgsz,
                batch   = batch,
                device  = device,
                verbose = False,
                plots   = False,
                project = str(out_dir / "eval"),
                name    = f"{variant}_{split}",
                exist_ok= True,
            )
            metrics = self._extractor.extract(result)
            self._log.log(
                f"[eval] {variant:<14s} | {split:<5s} | "
                f"P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"mAP50={metrics['mAP_50']:.4f}  "
                f"mAP50:95={metrics['mAP_50_95']:.4f}"
            )
            return metrics

        except Exception as e:
            self._log.log(f"[eval] {variant}/{split} FAILED: {e}")
            self._log.log(traceback.format_exc())
            return {"error": str(e), "mAP_50": 0.0, "mAP_50_95": 0.0}

    def evaluate_all(
        self,
        best_pt:     Path,
        yaml_path:   Path,
        out_dir:     Path,
        variant:     str,
        imgsz:       int,
        batch:       int,
        device_list: List[int],
        splits:      List[str] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate on ``splits`` (default: val + test) and return all results.

        Returns
        -------
        dict mapping split name → metrics dict
        """
        if splits is None:
            splits = ["val", "test"]

        device = device_list[0] if device_list else 0
        return {
            split: self.evaluate_split(
                best_pt, yaml_path, out_dir, variant, split, imgsz, batch, device
            )
            for split in splits
        }

    def __repr__(self) -> str:
        return (
            f"YoloEvaluator(classes={self._class_names})"
        )