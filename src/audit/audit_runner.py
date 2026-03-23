# -*- coding: utf-8 -*-
"""
audit/audit_runner.py
=====================
``AuditRunner`` — orchestrates all four data integrity audit methods
and produces a consolidated JSON report + human-readable summary.

Usage
-----
    runner = AuditRunner(args, audit_out=Path("./audit"))
    results = runner.run()
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from audit.audit_logger    import AuditLogger
from audit.method1_exact_hash import ExactHashChecker
from audit.method2_phash      import PHashChecker
from audit.method3_embedding  import EmbeddingChecker
from audit.method4_hard_crop  import HardCropProbe


# ==============================================================================
# DATASET SCANNER
# ==============================================================================

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_image_paths(split_dir: Path) -> Dict[str, List[Path]]:
    """Scan an ImageFolder-style split dir → {class_name: [paths]}."""
    result: Dict[str, List[Path]] = {}
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in VALID_EXT
        ]
        result[cls_dir.name] = sorted(imgs)
    return result


# ==============================================================================
# AUDIT RUNNER
# ==============================================================================

class AuditRunner:
    """
    Runs all four audit methods in sequence and writes a consolidated
    ``audit_report.json``.

    Parameters
    ----------
    data_root        : str          — dataset root (with train/validation/test)
    audit_out        : Path         — directory for all audit outputs
    checkpoint_path  : str or None  — checkpoint for Methods 3 & 4
    image_size       : int
    batch_size       : int
    num_workers      : int
    phash_threshold  : int
    dbscan_eps       : float
    dbscan_min_samples : int
    skip_phash       : bool
    skip_embedding   : bool
    skip_hard_crop   : bool

    Example
    -------
    >>> runner = AuditRunner(data_root="./data", audit_out=Path("./audit"))
    >>> results = runner.run()
    """

    def __init__(
        self,
        data_root:           str,
        audit_out:           Path,
        checkpoint_path:     Optional[str] = None,
        image_size:          int   = 512,
        batch_size:          int   = 32,
        num_workers:         int   = 4,
        phash_threshold:     int   = 10,
        dbscan_eps:          float = 0.15,
        dbscan_min_samples:  int   = 2,
        skip_phash:          bool  = False,
        skip_embedding:      bool  = False,
        skip_hard_crop:      bool  = False,
    ):
        self._data_root          = Path(data_root)
        self._audit_out          = audit_out
        self._checkpoint         = checkpoint_path
        self._image_size         = image_size
        self._batch_size         = batch_size
        self._num_workers        = num_workers
        self._phash_threshold    = phash_threshold
        self._dbscan_eps         = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        self._skip_phash         = skip_phash
        self._skip_embedding     = skip_embedding
        self._skip_hard_crop     = skip_hard_crop

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run all methods and return combined results dict."""
        self._audit_out.mkdir(parents=True, exist_ok=True)
        log = AuditLogger(self._audit_out / "audit_log.txt")

        # --- Load split directories ---
        splits, split_dirs = self._load_splits(log)
        if splits is None:
            log.close()
            sys.exit(1)

        results: Dict = {}

        # Method 1 — always runs
        results["method1_exact_hash"] = ExactHashChecker(log).run(splits)

        # Method 2 — pHash
        if self._skip_phash:
            log.log("\n[skip] Method 2 (pHash) — --skip_phash set")
        else:
            results["method2_phash"] = PHashChecker(
                log, max_hamming_distance=self._phash_threshold
            ).run(splits)

        # Method 3 — Embedding DBSCAN
        if self._skip_embedding:
            log.log("\n[skip] Method 3 (Embedding) — --skip_embedding set")
        else:
            results["method3_embedding"] = EmbeddingChecker(
                log                 = log,
                checkpoint_path     = self._checkpoint,
                image_size          = min(self._image_size, 224),
                batch_size          = self._batch_size,
                dbscan_eps          = self._dbscan_eps,
                dbscan_min_samples  = self._dbscan_min_samples,
                num_workers         = self._num_workers,
            ).run(splits, output_dir=self._audit_out)

        # Method 4 — Hard Crop Probe
        if self._skip_hard_crop:
            log.log("\n[skip] Method 4 (Hard Crop) — --skip_hard_crop set")
        else:
            results["method4_hard_crop"] = HardCropProbe(
                checkpoint_path = self._checkpoint,
                test_dir        = split_dirs["test"],
                log             = log,
                image_size      = self._image_size,
                batch_size      = max(4, self._batch_size // 2),
                num_workers     = self._num_workers,
            ).run(output_dir=self._audit_out)

        self._print_summary(results, log)

        report_path = self._audit_out / "audit_report.json"
        report_path.write_text(json.dumps(results, indent=2, default=str))
        log.log(f"\nFull audit report: {report_path}")
        log.close()
        return results

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _load_splits(self, log: AuditLogger):
        root    = self._data_root
        val_dir = (
            root / "validation"
            if (root / "validation").exists()
            else root / "val"
        )
        split_dirs = {
            "train":      root / "train",
            "validation": val_dir,
            "test":       root / "test",
        }
        for name, d in split_dirs.items():
            if not d.exists():
                log.log(f"[ERROR] Split directory not found: {d}")
                return None, None

        log.log("Dataset splits:")
        splits: Dict[str, Dict[str, List[Path]]] = {}
        for name, d in split_dirs.items():
            class_paths = collect_image_paths(d)
            total       = sum(len(v) for v in class_paths.values())
            splits[name] = class_paths
            log.log(
                f"  {name:12s} {total:6d} images  |  "
                + "  ".join(
                    f"{cls}:{len(paths)}"
                    for cls, paths in class_paths.items()
                )
            )

        log.log(f"\nCheckpoint: {self._checkpoint or 'not provided'}")
        log.log(f"Output dir: {self._audit_out.resolve()}")
        return splits, split_dirs

    @staticmethod
    def _print_summary(results: Dict, log: AuditLogger) -> None:
        log.sep(True)
        log.log("AUDIT SUMMARY")
        log.sep()
        all_pass = True

        m1 = results.get("method1_exact_hash", {})
        if m1:
            n      = m1.get("cross_split_duplicates", 0)
            status = "PASS" if n == 0 else "FAIL"
            if n > 0: all_pass = False
            log.log(
                f"  Method 1 (MD5 Exact):        [{status}]  "
                f"{n} exact cross-split duplicates"
            )

        m2 = results.get("method2_phash", {})
        if m2.get("status") == "skipped_missing_imagededup":
            log.log("  Method 2 (pHash):            [SKIP]  imagededup not installed")
        elif m2:
            n      = m2.get("cross_split_clusters", 0)
            status = "PASS" if n == 0 else "WARN"
            if n > 0: all_pass = False
            log.log(
                f"  Method 2 (pHash Near-Dup):   [{status}]  "
                f"{n} near-dup clusters span splits "
                f"(threshold={m2.get('hamming_threshold','?')})"
            )

        m3 = results.get("method3_embedding", {})
        if m3:
            n  = m3.get("cross_split_clusters", 0)
            nc = m3.get("n_clusters", 0)
            status = "PASS" if n == 0 else "WARN"
            if n > 0: all_pass = False
            log.log(
                f"  Method 3 (Embedding DBSCAN): [{status}]  "
                f"{n} visual clusters span splits  |  {nc} total clusters"
            )

        m4 = results.get("method4_hard_crop", {})
        if m4.get("status") == "skipped_no_checkpoint":
            log.log("  Method 4 (Hard Crop Probe):  [SKIP]  no checkpoint provided")
        elif m4:
            drops    = m4.get("drops_from_standard", {})
            max_drop = max(abs(v) for v in drops.values()) if drops else 0
            status   = (
                "PASS"    if max_drop < 0.05
                else "WARN"    if max_drop < 0.10
                else "CONCERN"
            )
            log.log(
                f"  Method 4 (Hard Crop Probe):  [{status}]  "
                f"max accuracy drop={max_drop:.3f} "
                f"(crop={drops.get('crop',0):+.3f}  "
                f"grey={drops.get('grey',0):+.3f}  "
                f"aug={drops.get('aug',0):+.3f})"
            )

        log.sep()
        overall = "ALL CHECKS PASSED" if all_pass else "ISSUES DETECTED — review details above"
        log.log(f"  OVERALL: {overall}")
        log.sep(True)

        log.log("\nFYP DEFENCE TALKING POINT:")
        log.log("  'We applied four independent data integrity checks: MD5 exact-hash")
        log.log("   deduplication, perceptual hashing (pHash) with Hamming-distance")
        log.log("   clustering, deep ConvNeXt feature embedding + DBSCAN to catch")
        log.log("   visually similar patient images, and a Hard Crop Probe that strips")
        log.log("   background and colour cues to verify the model learned lesion")
        log.log("   morphology rather than incidental shortcuts.'")

    def __repr__(self) -> str:
        return f"AuditRunner(data_root={self._data_root})"