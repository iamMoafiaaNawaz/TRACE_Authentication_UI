# -*- coding: utf-8 -*-
"""
audit/method1_exact_hash.py
============================
Method 1: Exact MD5 Hash Check

Detects byte-perfect duplicate files across train / val / test splits.
Even a single shared file is definitive proof of data leakage.

FYP defence talking point
--------------------------
"We computed MD5 checksums for every image across all three splits and
confirmed zero byte-identical files exist across split boundaries."
"""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm.auto import tqdm

from audit.audit_logger import AuditLogger


class ExactHashChecker:
    """
    Detects byte-perfect duplicate images across dataset splits via MD5.

    Parameters
    ----------
    log : AuditLogger

    Example
    -------
    >>> checker = ExactHashChecker(log)
    >>> result = checker.run(splits)
    >>> print(result["cross_split_duplicates"])
    """

    def __init__(self, log: AuditLogger):
        self._log = log

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(self, splits: Dict[str, Dict[str, List[Path]]]) -> Dict:
        """
        Compute MD5 for every image in every split and report cross-split
        hash collisions.

        Parameters
        ----------
        splits : dict of {split_name: {class_name: [image_paths]}}

        Returns
        -------
        dict with keys: method, cross_split_duplicates, examples
        """
        self._log.sep(True)
        self._log.log("METHOD 1: EXACT MD5 HASH CHECK")
        self._log.log("  Detects byte-perfect duplicate files across splits.")
        self._log.sep()

        hash_to_locations: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

        for split_name, class_paths in splits.items():
            all_paths = self._flat(class_paths)
            self._log.log(f"  Hashing {len(all_paths)} files in [{split_name}]...")
            for p in tqdm(all_paths, desc=f"  MD5 {split_name}", leave=False):
                h = self._md5(p)
                hash_to_locations[h].append((split_name, p))

        # Hashes that appear in MORE THAN ONE split
        cross_split = {
            h: locs for h, locs in hash_to_locations.items()
            if len({s for s, _ in locs}) > 1
        }

        result = {
            "method": "exact_md5",
            "cross_split_duplicates": len(cross_split),
            "examples": [],
        }

        if cross_split:
            self._log.log(
                f"\n  [FAIL] {len(cross_split)} exact duplicates found across splits!"
            )
            for h, locs in list(cross_split.items())[:10]:
                self._log.log(f"    Hash {h[:12]}...")
                for split, path in locs:
                    self._log.log(f"      [{split}] {path.name}")
                result["examples"].append({
                    "hash": h,
                    "locations": [(s, str(p)) for s, p in locs],
                })
        else:
            self._log.log(
                "\n  [PASS] No byte-identical files found across splits."
            )

        return result

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _flat(class_paths: Dict[str, List[Path]]) -> List[Path]:
        return [p for paths in class_paths.values() for p in paths]

    @staticmethod
    def _md5(path: Path, chunk: int = 65_536) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def __repr__(self) -> str:
        return "ExactHashChecker()"