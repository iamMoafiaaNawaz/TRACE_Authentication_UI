# -*- coding: utf-8 -*-
"""
evaluation/metrics.py
=====================
All metric computation for the TRACE pipeline.

Contains two layers:
1. **Pure functions** — stateless, take numpy arrays, return scalar values.
   Used directly in unit tests and by ``MetricsCalculator``.

2. **MetricsCalculator** — OOP wrapper that takes raw prediction arrays and
   returns the full metrics dict consumed by ``evaluate_model.py``.

Metrics computed
----------------
Classification (ConvNeXt)
    accuracy, balanced_accuracy, MCC, Cohen's kappa,
    macro P/R/F1, weighted P/R/F1,
    macro AUC-OvR, macro AUC-OvO, macro PR-AUC

Detection (YOLO)
    precision, recall, F1, mAP@0.5, mAP@0.5:0.95, per-class AP
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ==============================================================================
# PURE METRIC FUNCTIONS
# ==============================================================================

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard accuracy — fraction of correctly classified samples."""
    return float(accuracy_score(y_true, y_pred))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy — mean per-class recall.
    More informative than accuracy on imbalanced datasets.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient — single balanced score in [-1, 1].
    +1 = perfect prediction, 0 = random, -1 = inverse prediction.
    """
    return float(matthews_corrcoef(y_true, y_pred))


def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cohen's kappa — agreement beyond chance.
    Particularly useful for imbalanced multi-class problems.
    """
    return float(cohen_kappa_score(y_true, y_pred))


def compute_macro_prf1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Macro-averaged precision, recall, and F1.
    Equal weight per class — appropriate for class-imbalanced datasets.
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "macro_precision": float(p),
        "macro_recall":    float(r),
        "macro_f1":        float(f1),
    }


def compute_weighted_prf1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Weighted-average precision, recall, and F1.
    Weights by support (number of true instances per class).
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "weighted_precision": float(p),
        "weighted_recall":    float(r),
        "weighted_f1":        float(f1),
    }


def compute_auc_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, Optional[float]]:
    """
    Compute macro AUC-OvR, AUC-OvO, and mean PR-AUC.

    Returns ``None`` for each metric if computation fails (e.g. only one
    class present in ``y_true`` for a small split).
    """
    result: Dict[str, Optional[float]] = {
        "macro_auc_ovr": None,
        "macro_auc_ovo": None,
        "macro_pr_auc":  None,
    }
    try:
        yoh = np.eye(num_classes)[y_true.astype(int)]
        result["macro_auc_ovr"] = float(
            roc_auc_score(yoh, y_prob, average="macro", multi_class="ovr")
        )
        result["macro_auc_ovo"] = float(
            roc_auc_score(yoh, y_prob, average="macro", multi_class="ovo")
        )
        pr_aucs = []
        for i in range(num_classes):
            prec_c, rec_c, _ = precision_recall_curve(yoh[:, i], y_prob[:, i])
            pr_aucs.append(auc(rec_c, prec_c))
        result["macro_pr_auc"] = float(np.mean(pr_aucs))
    except Exception:
        pass
    return result


def compute_f1_from_pr(precision: float, recall: float) -> float:
    """F1 from scalar precision and recall — used by YOLO metrics."""
    return 2 * precision * recall / (precision + recall + 1e-8)


# ==============================================================================
# METRICS CALCULATOR  (OOP wrapper for ConvNeXt evaluation)
# ==============================================================================

class MetricsCalculator:
    """
    Computes the full classification metrics dict from raw prediction arrays.

    Used by ``ConvNextEvaluator.evaluate()`` after collecting all predictions.

    Parameters
    ----------
    num_classes : int

    Example
    -------
    >>> calc = MetricsCalculator(num_classes=4)
    >>> metrics = calc.compute(
    ...     y_true=np.array([0,1,2,3]),
    ...     y_pred=np.array([0,1,3,3]),
    ...     y_prob=np.random.rand(4,4),
    ...     total_loss=0.45,
    ...     n_samples=4,
    ... )
    >>> print(metrics["macro_f1"])
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def compute(
        self,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
        y_prob:     np.ndarray,
        total_loss: float,
        n_samples:  int,
    ) -> Dict:
        """
        Compute all metrics and return as a flat dict.

        Returned keys
        -------------
        loss, acc, balanced_acc, mcc, kappa,
        macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1,
        macro_auc_ovr, macro_auc_ovo, macro_pr_auc,
        preds, labels, probs
        """
        metrics: Dict = {}

        metrics["loss"]          = total_loss / max(n_samples, 1)
        metrics["acc"]           = compute_accuracy(y_true, y_pred)
        metrics["balanced_acc"]  = compute_balanced_accuracy(y_true, y_pred)
        metrics["mcc"]           = compute_mcc(y_true, y_pred)
        metrics["kappa"]         = compute_kappa(y_true, y_pred)

        metrics.update(compute_macro_prf1(y_true, y_pred))
        metrics.update(compute_weighted_prf1(y_true, y_pred))
        metrics.update(compute_auc_scores(y_true, y_prob, self.num_classes))

        # Raw arrays — kept for downstream plots / confusion matrices
        metrics["preds"]  = y_pred.tolist()
        metrics["labels"] = y_true.tolist()
        metrics["probs"]  = y_prob.tolist()

        return metrics

    def __repr__(self) -> str:
        return f"MetricsCalculator(num_classes={self.num_classes})"


# ==============================================================================
# YOLO METRICS EXTRACTOR
# ==============================================================================

class YoloMetricsExtractor:
    """
    Extracts a standardised metrics dict from an ultralytics validation result.

    Handles missing attributes gracefully — returns 0.0 for any field that
    cannot be read from the result object.

    Example
    -------
    >>> extractor = YoloMetricsExtractor(class_names=["BCC","BKL","MEL","NV"])
    >>> metrics = extractor.extract(ultralytics_result)
    >>> print(metrics["mAP_50"])
    """

    def __init__(self, class_names: List[str]):
        self._class_names = class_names

    def extract(self, result) -> Dict:
        """
        Extract metrics from an ultralytics ``Results`` object.

        Returns
        -------
        dict with keys:
            precision, recall, f1, mAP_50, mAP_50_95, per_class_AP
        """
        box = getattr(result, "box", None)

        mp  = float(getattr(box, "mp",    0.0)) if box else 0.0
        mr  = float(getattr(box, "mr",    0.0)) if box else 0.0
        m50 = float(getattr(box, "map50", 0.0)) if box else 0.0
        m95 = float(getattr(box, "map",   0.0)) if box else 0.0
        f1  = compute_f1_from_pr(mp, mr)

        per_class: Dict[str, float] = {}
        if box and hasattr(box, "ap_class_index") and hasattr(box, "ap"):
            for i, ci in enumerate(box.ap_class_index):
                name = (
                    self._class_names[int(ci)]
                    if int(ci) < len(self._class_names)
                    else str(ci)
                )
                per_class[name] = float(box.ap[i]) if i < len(box.ap) else 0.0

        return {
            "precision":    mp,
            "recall":       mr,
            "f1":           f1,
            "mAP_50":       m50,
            "mAP_50_95":    m95,
            "per_class_AP": per_class,
        }

    def __repr__(self) -> str:
        return f"YoloMetricsExtractor(classes={self._class_names})"