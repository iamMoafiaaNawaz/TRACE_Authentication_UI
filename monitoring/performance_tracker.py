# -*- coding: utf-8 -*-
"""
monitoring/performance_tracker.py
===================================
Runtime performance monitoring for the TRACE clinical deployment.

Tracks per-prediction metrics (confidence distribution, class frequency,
latency) and flags statistical drift from the training-time distribution.

Classes
-------
PredictionLogger
    Appends every inference result to a rolling JSONL log.

DriftDetector
    Compares live confidence distribution against training-time baseline
    and raises a DriftWarning when divergence exceeds threshold.

PerformanceSummary
    Generates a daily summary report from the JSONL log.

Usage
-----
    from monitoring.performance_tracker import PredictionLogger, DriftDetector

    logger   = PredictionLogger(log_dir=Path("./monitoring/logs"))
    detector = DriftDetector(baseline_path=Path("./monitoring/baseline.json"))

    result = pipeline.predict_image(img_path)
    logger.log(result)
    detector.check(result)
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# ==============================================================================
# PREDICTION LOGGER
# ==============================================================================

class PredictionLogger:
    """
    Appends every inference result to a rolling JSONL log file.

    Each line is a JSON object with timestamp, prediction, confidence,
    and full probability distribution.

    Parameters
    ----------
    log_dir : Path — directory where JSONL logs are written
    rotate_daily : bool — start a new file each day (default True)

    Example
    -------
    >>> logger = PredictionLogger(Path("./monitoring/logs"))
    >>> logger.log({"prediction": "MEL", "confidence": 0.87, "class_probs": {...}})
    """

    def __init__(self, log_dir: Path, rotate_daily: bool = True):
        self._log_dir     = log_dir
        self._rotate_daily = rotate_daily
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, result: Dict, image_path: Optional[str] = None) -> None:
        """Append a single inference result to the current log file."""
        entry = {
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "image":       str(image_path) if image_path else None,
            "prediction":  result.get("prediction"),
            "confidence":  result.get("confidence"),
            "class_probs": result.get("class_probs", {}),
        }
        log_path = self._current_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _current_log_path(self) -> Path:
        suffix = time.strftime("%Y-%m-%d") if self._rotate_daily else "all"
        return self._log_dir / f"predictions_{suffix}.jsonl"

    def __repr__(self) -> str:
        return f"PredictionLogger(log_dir={self._log_dir})"


# ==============================================================================
# DRIFT DETECTOR
# ==============================================================================

class DriftWarning(UserWarning):
    """Raised when live prediction distribution drifts from baseline."""


class DriftDetector:
    """
    Monitors confidence and class frequency distributions for drift.

    Compares a rolling window of live predictions against a training-time
    baseline. Emits a DriftWarning when Jensen-Shannon divergence exceeds
    the configured threshold.

    Parameters
    ----------
    baseline_path : Path — JSON file with baseline class frequencies
    window_size   : int  — number of recent predictions to compare (default 100)
    js_threshold  : float — JS divergence threshold for drift alert (default 0.1)

    Example
    -------
    >>> detector = DriftDetector(Path("./monitoring/baseline.json"))
    >>> detector.check({"prediction": "MEL", "confidence": 0.87})
    """

    def __init__(
        self,
        baseline_path: Path,
        window_size:   int   = 100,
        js_threshold:  float = 0.10,
    ):
        self._window_size  = window_size
        self._js_threshold = js_threshold
        self._window:      List[str]  = []
        self._baseline: Dict[str, float] = {}

        if baseline_path.exists():
            self._baseline = json.loads(baseline_path.read_text())

    def check(self, result: Dict) -> None:
        """Add result to rolling window and check for drift."""
        import warnings
        pred = result.get("prediction")
        if not pred:
            return

        self._window.append(pred)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        if len(self._window) >= self._window_size and self._baseline:
            js = self._js_divergence()
            if js > self._js_threshold:
                warnings.warn(
                    f"[DriftDetector] JS divergence={js:.4f} > threshold={self._js_threshold}. "
                    f"Live class distribution may have drifted from training baseline. "
                    f"Review recent predictions.",
                    DriftWarning,
                    stacklevel=2,
                )

    def save_baseline(self, log_dir: Path, baseline_path: Path) -> None:
        """
        Compute a new baseline from all JSONL logs in log_dir and save.
        Run once after a stable deployment period.
        """
        counter: Counter = Counter()
        for jsonl in sorted(log_dir.glob("predictions_*.jsonl")):
            for line in jsonl.read_text().splitlines():
                try:
                    entry = json.loads(line)
                    if entry.get("prediction"):
                        counter[entry["prediction"]] += 1
                except Exception:
                    pass

        total = sum(counter.values())
        if total == 0:
            return

        baseline = {cls: count / total for cls, count in counter.items()}
        baseline_path.write_text(json.dumps(baseline, indent=2))
        print(f"[DriftDetector] Baseline saved: {baseline_path}  (n={total})")

    def _js_divergence(self) -> float:
        """Jensen-Shannon divergence between live window and baseline."""
        import math
        live_counter = Counter(self._window)
        total        = len(self._window)
        classes      = sorted(set(list(self._baseline.keys()) + list(live_counter.keys())))
        eps          = 1e-10

        p = [self._baseline.get(c, eps) for c in classes]
        q = [live_counter.get(c, 0) / total for c in classes]
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        def kl(a, b):
            return sum(ai * math.log(ai / bi + eps) for ai, bi in zip(a, b) if ai > 0)

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    def __repr__(self) -> str:
        return (
            f"DriftDetector("
            f"window={self._window_size}, "
            f"threshold={self._js_threshold})"
        )


# ==============================================================================
# PERFORMANCE SUMMARY
# ==============================================================================

class PerformanceSummary:
    """
    Generates a daily summary report from JSONL prediction logs.

    Reports: total predictions, class distribution, mean/min/max confidence,
    low-confidence rate (< 0.5), and high-risk class frequency.

    Parameters
    ----------
    log_dir : Path — directory containing prediction JSONL files

    Example
    -------
    >>> summary = PerformanceSummary(Path("./monitoring/logs"))
    >>> report = summary.generate(date="2026-03-18")
    >>> print(report["total_predictions"])
    """

    HIGH_RISK_CLASSES = {"MEL", "BCC"}
    LOW_CONF_THRESHOLD = 0.50

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir

    def generate(self, date: Optional[str] = None) -> Dict:
        """
        Generate a summary for a specific date (YYYY-MM-DD).
        If date is None, summarises all logs in log_dir.
        """
        entries = self._load(date)
        if not entries:
            return {"date": date, "total_predictions": 0, "note": "No predictions logged"}

        confs       = [e["confidence"] for e in entries if e.get("confidence") is not None]
        predictions = [e["prediction"]  for e in entries if e.get("prediction")]
        class_dist  = dict(Counter(predictions))
        n           = len(predictions)

        return {
            "date":                date or "all",
            "total_predictions":   n,
            "class_distribution":  class_dist,
            "class_frequency_pct": {k: round(v/n*100, 1) for k, v in class_dist.items()},
            "confidence": {
                "mean":  round(sum(confs) / len(confs), 4) if confs else None,
                "min":   round(min(confs), 4) if confs else None,
                "max":   round(max(confs), 4) if confs else None,
            },
            "low_confidence_rate": (
                round(sum(1 for c in confs if c < self.LOW_CONF_THRESHOLD) / len(confs), 4)
                if confs else None
            ),
            "high_risk_predictions": sum(
                1 for p in predictions if p in self.HIGH_RISK_CLASSES
            ),
            "high_risk_rate": round(
                sum(1 for p in predictions if p in self.HIGH_RISK_CLASSES) / n, 4
            ) if n else 0,
        }

    def _load(self, date: Optional[str]) -> List[Dict]:
        entries = []
        pattern = f"predictions_{date}.jsonl" if date else "predictions_*.jsonl"
        for jsonl in sorted(self._log_dir.glob(pattern)):
            for line in jsonl.read_text().splitlines():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return entries

    def __repr__(self) -> str:
        return f"PerformanceSummary(log_dir={self._log_dir})"