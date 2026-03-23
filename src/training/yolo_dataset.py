# -*- coding: utf-8 -*-
"""
src/training/yolo_dataset.py
=============================
Dataset utilities for the YOLO localization pipeline.

Classes / Functions
-------------------
scan_class_folder   — scans an ImageFolder-style directory into records
load_splits         — loads pre-split or auto-splits from raw class folders
YoloDatasetBuilder  — converts classification records into a YOLO dataset
                      (images + label .txt files + dataset.yaml)
"""

import collections
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.io_ops import LiveLogger


# ==============================================================================
# CONSTANTS
# ==============================================================================

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ==============================================================================
# DATASET SCANNER
# ==============================================================================

def scan_class_folder(
    root: Path,
) -> Tuple[List[Tuple[Path, str, int]], List[str]]:
    """
    Scan an ImageFolder-style directory.

    Returns
    -------
    records : List of (image_path, class_name, class_index)
    names   : Sorted list of class names
    """
    root = Path(root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not dirs:
        raise ValueError(f"No class subdirectories found in: {root}")

    names: List[str] = [d.name for d in dirs]
    records: List[Tuple[Path, str, int]] = []
    seen: set = set()

    for idx, d in enumerate(dirs):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VALID_EXT and p not in seen:
                records.append((p, names[idx], idx))
                seen.add(p)

    return records, names


# ==============================================================================
# SPLIT LOADER
# ==============================================================================

def load_splits(
    split_root: Path,
    data_root:  Path,
    train_frac: float,
    val_frac:   float,
    seed:       int,
) -> Tuple[
    List[Tuple[Path, str, int]],
    List[Tuple[Path, str, int]],
    List[Tuple[Path, str, int]],
    List[str],
]:
    """
    Load train / val / test records.

    Priority:
    1. Pre-split dirs under ``split_root`` (train/ + validation/ + test/)
    2. Auto-split from ``data_root`` using ``train_frac`` / ``val_frac``

    Returns
    -------
    train_records, val_records, test_records, class_names
    """
    sp = Path(split_root)
    vn = "validation" if (sp / "validation").exists() else "val"

    if all((sp / d).exists() for d in ["train", vn, "test"]):
        tr,    names = scan_class_folder(sp / "train")
        va,    _     = scan_class_folder(sp / vn)
        te,    _     = scan_class_folder(sp / "test")
        return tr, va, te, names

    # Auto-split from raw data_root
    all_r, names = scan_class_folder(Path(data_root))
    by: Dict[int, List] = collections.defaultdict(list)
    for r in all_r:
        by[r[2]].append(r)

    tr, va, te = [], [], []
    rng = random.Random(seed)
    for recs in by.values():
        rng.shuffle(recs)
        n  = len(recs)
        n1 = max(1, int(n * train_frac))
        n2 = max(n1 + 1, int(n * (train_frac + val_frac)))
        tr.extend(recs[:n1])
        va.extend(recs[n1:n2])
        te.extend(recs[n2:])

    return tr, va, te, names


# ==============================================================================
# YOLO DATASET BUILDER
# ==============================================================================

class YoloDatasetBuilder:
    """
    Converts classification records into a YOLO-format dataset.

    For each image:
    - Creates a symlink (or copy) under ``images/<split>/``
    - Writes a ``labels/<split>/<stem>.txt`` with the pseudo bounding box

    Also writes a ``dataset.yaml`` compatible with ``ultralytics``.

    Parameters
    ----------
    yolo_root    : Path          — destination root for the YOLO dataset
    gen          : PseudoBoxGenerator — box generator (otsu or locmap mode)
    log          : LiveLogger
    copy_images  : bool          — copy images instead of symlinking

    Example
    -------
    >>> builder = YoloDatasetBuilder(Path("./yolo_ds"), gen, log)
    >>> yaml_path = builder.build(
    ...     splits={"train": tr, "val": va, "test": te},
    ...     names=["BCC", "BKL", "MEL", "NV"],
    ... )
    """

    def __init__(
        self,
        yolo_root:   Path,
        gen,                       # PseudoBoxGenerator
        log:         LiveLogger,
        copy_images: bool = False,
    ):
        self._root        = Path(yolo_root)
        self._gen         = gen
        self._log         = log
        self._copy_images = copy_images

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def build(
        self,
        splits: Dict[str, List[Tuple[Path, str, int]]],
        names:  List[str],
    ) -> Path:
        """
        Build the full YOLO dataset directory and return the path to
        ``dataset.yaml``.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        self._log.log(
            f"[dataset] Building in: {self._root}  mode={self._gen.mode}"
        )
        total = 0

        for split_name, records in splits.items():
            total += self._build_split(split_name, records)

        yaml_path = self._write_yaml(names)
        self._log.log(f"[dataset] Labels: {total}  yaml: {yaml_path}")
        return yaml_path

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _build_split(
        self, split_name: str, records: List[Tuple[Path, str, int]]
    ) -> int:
        idir = self._root / "images" / split_name
        ldir = self._root / "labels" / split_name
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        self._log.log(f"[dataset] {split_name}: {len(records)} images")

        for i, (src, cls_name, cls_idx) in enumerate(records):
            stem = f"{i:06d}_{src.stem[:40]}"
            dst  = idir / (stem + src.suffix.lower())
            lbl  = ldir / (stem + ".txt")

            # Image — symlink preferred, copy as fallback
            if not dst.exists():
                if self._copy_images:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.symlink_to(src.resolve())
                    except (OSError, NotImplementedError):
                        shutil.copy2(src, dst)

            # Label
            cx, cy, bw, bh = self._gen.get_box(src)
            lbl.write_text(
                f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

            if (i + 1) % 1000 == 0:
                self._log.log(
                    f"  [dataset] {split_name} {i + 1}/{len(records)}"
                )

        return len(records)

    def _write_yaml(self, names: List[str]) -> Path:
        yaml_path = self._root / "dataset.yaml"
        yaml_path.write_text(
            f"path: {self._root.resolve()}\n"
            f"train: images/train\n"
            f"val:   images/val\n"
            f"test:  images/test\n"
            f"nc: {len(names)}\n"
            f"names:\n"
            + "".join(f"  - {n}\n" for n in names)
        )
        return yaml_path

    def __repr__(self) -> str:
        return (
            f"YoloDatasetBuilder("
            f"root={self._root}, "
            f"mode={self._gen.mode}, "
            f"copy={self._copy_images})"
        )