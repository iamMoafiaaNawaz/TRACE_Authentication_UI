# -*- coding: utf-8 -*-
"""
audit/method3_embedding.py
===========================
Method 3: Deep Feature Embedding + DBSCAN Clustering

Runs all images through a frozen ConvNeXt-Base backbone to get 1024-dim
L2-normalised embeddings, then clusters with DBSCAN.

Catches same-patient images with different framing/lighting that pHash
misses, because the model's feature space is semantic — not pixel-level.

Optionally saves a UMAP 2D scatter plot of the embedding space.

FYP defence talking point
--------------------------
"We extracted 1024-dim ConvNeXt-Base embeddings for every image, applied
DBSCAN clustering in L2-normalised feature space, and verified that no
visually similar image clusters cross split boundaries."
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from tqdm.auto import tqdm

from audit.audit_logger import AuditLogger


# ==============================================================================
# MINIMAL IMAGE DATASET (no class structure needed)
# ==============================================================================

class _ImageListDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception:
            return torch.zeros(3, 224, 224), idx


# ==============================================================================
# EMBEDDING CHECKER
# ==============================================================================

class EmbeddingChecker:
    """
    Deep feature embedding + DBSCAN leakage detection.

    Uses a frozen ConvNeXt-Base backbone (ImageNet weights or provided
    checkpoint) to extract 1024-dim embeddings, then clusters with DBSCAN.

    Parameters
    ----------
    log              : AuditLogger
    checkpoint_path  : str or None — optional checkpoint to load backbone weights
    image_size       : int         — resize target (default 224)
    batch_size       : int
    dbscan_eps       : float       — cosine distance threshold
    dbscan_min_samples : int
    num_workers      : int

    Example
    -------
    >>> checker = EmbeddingChecker(log, checkpoint_path="./best.pth")
    >>> result = checker.run(splits, output_dir=Path("./audit"))
    """

    EMBED_DIM = 1024

    def __init__(
        self,
        log:                AuditLogger,
        checkpoint_path:    Optional[str] = None,
        image_size:         int  = 224,
        batch_size:         int  = 32,
        dbscan_eps:         float = 0.15,
        dbscan_min_samples: int  = 2,
        num_workers:        int  = 4,
    ):
        self._log          = log
        self._ckpt         = checkpoint_path
        self._image_size   = image_size
        self._batch_size   = batch_size
        self._eps          = dbscan_eps
        self._min_samples  = dbscan_min_samples
        self._num_workers  = num_workers

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        splits:     Dict[str, Dict[str, List[Path]]],
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Extract embeddings, cluster with DBSCAN, report cross-split clusters.
        Optionally save a UMAP 2D scatter plot.
        """
        self._log.sep(True)
        self._log.log("METHOD 3: DEEP FEATURE EMBEDDING + DBSCAN CLUSTERING")
        self._log.log(
            f"  DBSCAN eps={self._eps}  min_samples={self._min_samples}"
        )
        self._log.log(
            "  Catches visually similar images that pHash misses."
        )
        self._log.sep()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log.log(f"  Device: {device}")

        extractor = self._build_extractor(device)

        all_paths, path_split = self._collect(splits)
        self._log.log(f"  Total images: {len(all_paths)}")

        embeddings = self._extract_embeddings(extractor, all_paths, device)
        emb_normed = normalize(embeddings, norm="l2")
        self._log.log(f"  Embeddings shape: {emb_normed.shape}")

        labels     = self._dbscan(emb_normed)
        cross_split = self._find_cross_split(labels, all_paths, path_split)

        result = {
            "method":               "embedding_dbscan",
            "dbscan_eps":           self._eps,
            "dbscan_min_samples":   self._min_samples,
            "total_images":         len(all_paths),
            "n_clusters":           len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise":              int((labels == -1).sum()),
            "cross_split_clusters": len(cross_split),
            "details":              list(cross_split.values())[:50],
        }

        if cross_split:
            self._log.log(
                f"\n  [WARN] {len(cross_split)} visual clusters span multiple splits!"
            )
            for info in list(cross_split.values())[:5]:
                self._log.log(
                    f"    Cluster {info['cluster_id']} ({info['size']} images) "
                    f"spans: {info['splits']}"
                )
                for split, path in info["members"][:3]:
                    self._log.log(f"      [{split}] {Path(path).name}")
        else:
            self._log.log(
                f"\n  [PASS] No visual clusters span multiple splits "
                f"(eps={self._eps})."
            )

        if output_dir:
            self._maybe_umap(emb_normed, path_split, output_dir)

        return result

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _build_extractor(self, device: torch.device) -> nn.Module:
        self._log.log("  Loading ConvNeXt-Base feature extractor...")
        backbone  = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(1),
        ).to(device).eval()

        if self._ckpt and Path(self._ckpt).exists():
            try:
                ck = torch.load(self._ckpt, map_location=device, weights_only=False)
                sd = ck.get("model_state_dict", ck)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                fe_sd = {
                    k: v for k, v in sd.items()
                    if k.startswith("features.") or k.startswith("avgpool.")
                }
                if fe_sd:
                    extractor.load_state_dict(fe_sd, strict=False)
                    self._log.log(
                        f"  Loaded backbone weights from: {Path(self._ckpt).name}"
                    )
                else:
                    self._log.log("  Could not extract backbone keys — using ImageNet weights.")
            except Exception as e:
                self._log.log(f"  [warn] Checkpoint load failed ({e}) — using ImageNet weights.")
        else:
            self._log.log("  No checkpoint — using ImageNet pretrained weights.")

        return extractor

    def _extract_embeddings(
        self,
        extractor: nn.Module,
        all_paths: List[Path],
        device:    torch.device,
    ) -> np.ndarray:
        tf = transforms.Compose([
            transforms.Resize(self._image_size + 32),
            transforms.CenterCrop(self._image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        ds     = _ImageListDataset(all_paths, tf)
        loader = DataLoader(
            ds, batch_size=self._batch_size, shuffle=False,
            num_workers=self._num_workers,
            pin_memory=(device.type == "cuda"),
        )
        self._log.log("  Extracting embeddings (may take a few minutes)...")
        embeddings = np.zeros((len(all_paths), self.EMBED_DIM), dtype=np.float32)
        with torch.no_grad():
            for imgs, idxs in tqdm(loader, desc="  Embeddings", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                feats = extractor(imgs).cpu().numpy()
                for feat, orig_idx in zip(feats, idxs.numpy()):
                    embeddings[orig_idx] = feat
        return embeddings

    def _dbscan(self, emb_normed: np.ndarray) -> np.ndarray:
        self._log.log(
            f"  Running DBSCAN (eps={self._eps}, min_samples={self._min_samples})..."
        )
        t0 = time.time()
        db = DBSCAN(
            eps=self._eps, min_samples=self._min_samples,
            metric="euclidean", n_jobs=-1,
        )
        labels = db.fit_predict(emb_normed)
        self._log.log(
            f"  DBSCAN finished in {time.time()-t0:.1f}s  |  "
            f"clusters={len(set(labels)) - (1 if -1 in labels else 0)}  "
            f"noise={int((labels == -1).sum())}"
        )
        return labels

    def _find_cross_split(
        self,
        labels:    np.ndarray,
        all_paths: List[Path],
        path_split: List[str],
    ) -> Dict:
        cluster_splits:  Dict[int, Set[str]]         = defaultdict(set)
        cluster_members: Dict[int, List]              = defaultdict(list)
        for idx, (lbl, split) in enumerate(zip(labels, path_split)):
            if lbl == -1:
                continue
            cluster_splits[lbl].add(split)
            cluster_members[lbl].append((split, str(all_paths[idx])))

        return {
            lbl: {
                "cluster_id": lbl,
                "splits":     sorted(cluster_splits[lbl]),
                "size":       len(cluster_members[lbl]),
                "members":    cluster_members[lbl][:8],
            }
            for lbl, sp in cluster_splits.items()
            if len(sp) > 1
        }

    def _maybe_umap(
        self,
        emb_normed: np.ndarray,
        path_split: List[str],
        output_dir: Path,
    ) -> None:
        try:
            import umap
            import matplotlib.pyplot as plt
        except ImportError:
            self._log.log("  [skip] umap-learn not installed — no UMAP plot.")
            return

        if len(emb_normed) > 50_000:
            self._log.log("  [skip] Too many images for UMAP (>50k).")
            return

        self._log.log("  Generating UMAP 2D scatter plot...")
        try:
            reducer = umap.UMAP(
                n_components=2, n_neighbors=15,
                min_dist=0.1, random_state=42, n_jobs=-1,
            )
            emb2d       = reducer.fit_transform(emb_normed)
            split_names = sorted(set(path_split))
            colours     = plt.cm.Set1(np.linspace(0, 1, len(split_names)))
            split_colour = dict(zip(split_names, colours))

            fig, ax = plt.subplots(figsize=(10, 8))
            for sn in split_names:
                mask = np.array([s == sn for s in path_split])
                ax.scatter(
                    emb2d[mask, 0], emb2d[mask, 1],
                    c=[split_colour[sn]], s=2, alpha=0.4,
                    label=sn, rasterized=True,
                )
            ax.legend(markerscale=5)
            ax.set_title("UMAP of Image Embeddings by Split")
            ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
            plt.tight_layout()
            out = output_dir / "umap_embeddings.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            self._log.log(f"  UMAP plot saved: {out}")
        except Exception as e:
            self._log.log(f"  [warn] UMAP failed: {e}")

    @staticmethod
    def _collect(splits):
        all_paths  = []
        path_split = []
        for split_name, class_paths in splits.items():
            for paths in class_paths.values():
                for p in paths:
                    all_paths.append(p)
                    path_split.append(split_name)
        return all_paths, path_split

    def __repr__(self) -> str:
        return (
            f"EmbeddingChecker("
            f"eps={self._eps}, "
            f"min_samples={self._min_samples})"
        )