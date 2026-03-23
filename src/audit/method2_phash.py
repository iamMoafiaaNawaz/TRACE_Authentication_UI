# -*- coding: utf-8 -*-
"""
audit/method2_phash.py
=======================
Method 2: Perceptual Hash (pHash) Near-Duplicate Detection

Catches the same ISIC lesion photographed multiple times across splits
where MD5 would miss it due to resize / compression / minor rotation.

Uses ``imagededup`` PHash + Union-Find clustering on Hamming distance.
Gracefully skips if ``imagededup`` is not installed.

FYP defence talking point
--------------------------
"We applied perceptual hashing (pHash) with a Hamming-distance threshold
of 10 to detect near-duplicate images — e.g. the same lesion photographed
twice with slightly different framing — and confirmed no such clusters
span multiple splits."
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from audit.audit_logger import AuditLogger


class PHashChecker:
    """
    Near-duplicate detection using perceptual hashing (pHash).

    Requires ``imagededup`` (``pip install imagededup``).
    Gracefully skips if the library is not available.

    Parameters
    ----------
    log                  : AuditLogger
    max_hamming_distance : int   — lower = stricter (default 10)

    Example
    -------
    >>> checker = PHashChecker(log, max_hamming_distance=10)
    >>> result = checker.run(splits)
    """

    def __init__(self, log: AuditLogger, max_hamming_distance: int = 10):
        self._log      = log
        self._threshold = max_hamming_distance

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(self, splits: Dict[str, Dict[str, List[Path]]]) -> Dict:
        """
        Fingerprint all images with pHash and find near-duplicate clusters
        that span multiple splits.

        Returns
        -------
        dict with keys: method, hamming_threshold, total_images,
                        total_clusters_found, cross_split_clusters, details
        """
        self._log.sep(True)
        self._log.log("METHOD 2: PERCEPTUAL HASH (pHash) NEAR-DUPLICATE DETECTION")
        self._log.log(
            f"  Hamming distance threshold: {self._threshold}  (lower = stricter)"
        )
        self._log.log(
            "  Catches same lesion with minor resize/compression/rotation."
        )
        self._log.sep()

        try:
            from imagededup.methods import PHash
        except ImportError:
            self._log.log("  [SKIP] imagededup not installed.")
            self._log.log("         Install: pip install imagededup")
            return {"method": "phash", "status": "skipped_missing_imagededup"}

        phasher = PHash()

        # Build encoding map: unique key → pHash, avoiding filename collisions
        all_paths, path_to_split = self._collect(splits)
        self._log.log(f"  Total images to fingerprint: {len(all_paths)}")
        self._log.log("  Computing perceptual hashes...")

        encoding_map: Dict[str, str] = {}
        path_map:     Dict[str, Path] = {}

        for p in tqdm(all_paths, desc="  pHash", leave=False):
            split = path_to_split[str(p)]
            key   = f"{split}__{p.parent.name}__{p.name}"
            try:
                enc = phasher.encode_image(image_file=str(p))
                encoding_map[key] = enc
                path_map[key]     = p
            except Exception as e:
                self._log.log(f"  [warn] Could not hash {p.name}: {e}")

        self._log.log(f"  Encoded {len(encoding_map)} images successfully.")
        self._log.log("  Finding near-duplicate clusters...")

        duplicates = phasher.find_duplicates(
            encoding_map=encoding_map,
            max_distance_threshold=self._threshold,
            scores=True,
        )

        # Union-Find clustering
        parent = {k: k for k in encoding_map}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for img_key, similar_list in duplicates.items():
            for similar_key, _ in similar_list:
                union(img_key, similar_key)

        clusters: Dict[str, List[str]] = defaultdict(list)
        for k in encoding_map:
            clusters[find(k)].append(k)

        # Cross-split clusters
        cross_split_clusters = []
        for root, members in clusters.items():
            if len(members) < 2:
                continue
            member_splits = {k.split("__")[0] for k in members}
            if len(member_splits) > 1:
                cross_split_clusters.append({
                    "cluster_size":    len(members),
                    "splits_spanned":  sorted(member_splits),
                    "members": [
                        {
                            "key":   k,
                            "split": k.split("__")[0],
                            "file":  path_map[k].name,
                        }
                        for k in members
                    ],
                })

        result = {
            "method":               "phash",
            "hamming_threshold":    self._threshold,
            "total_images":         len(all_paths),
            "total_clusters_found": len(
                [c for c in clusters.values() if len(c) >= 2]
            ),
            "cross_split_clusters": len(cross_split_clusters),
            "details":              cross_split_clusters[:50],
        }

        if cross_split_clusters:
            self._log.log(
                f"\n  [WARN] {len(cross_split_clusters)} near-duplicate clusters span "
                "multiple splits!"
            )
            for cluster in cross_split_clusters[:5]:
                self._log.log(
                    f"    Cluster ({cluster['cluster_size']} images) "
                    f"spans: {cluster['splits_spanned']}"
                )
                for m in cluster["members"][:4]:
                    self._log.log(f"      [{m['split']}] {m['file']}")
        else:
            self._log.log(
                f"\n  [PASS] No near-duplicate clusters span multiple splits "
                f"(threshold={self._threshold})."
            )

        self._log.log(
            f"  Total near-duplicate groups (within any split): "
            f"{result['total_clusters_found']}"
        )
        return result

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _collect(splits):
        all_paths   = []
        path_to_split = {}
        for split_name, class_paths in splits.items():
            for paths in class_paths.values():
                for p in paths:
                    all_paths.append(p)
                    path_to_split[str(p)] = split_name
        return all_paths, path_to_split

    def __repr__(self) -> str:
        return f"PHashChecker(threshold={self._threshold})"