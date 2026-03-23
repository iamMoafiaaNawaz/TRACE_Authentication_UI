# coding: ascii
# =============================================================================
# exp8_localization.py  --  YOLO Skin Lesion Localization (thin entry point)
# =============================================================================
#
# All implementation lives in src/ and evaluation/.
# This file contains only argparse, lightweight helpers, and orchestration.
#
# TRAINING USAGE:
#   python exp8_localization.py                        # train all 3 models
#   python exp8_localization.py --single_gpu           # bypass DDP
#   python exp8_localization.py --skip_build           # reuse existing dataset
#
# MEDGEMMA API USAGE (after training, separate terminal / SLURM job):
#   python exp8_localization.py --serve_medgemma
#   python exp8_localization.py --serve_medgemma --medgemma_port 8787
#
# DOCTOR CALLS THE API:
#   POST http://localhost:8787/report
#     image=<file>  pred_class=MEL  pred_conf=0.87
#     box_cx=0.52   box_cy=0.48    box_w=0.31  box_h=0.28
#
# MODULE MAP:
#   src/utils/nms_patch.py        -- NMS disk + runtime patch
#   src/utils/io_ops.py           -- LiveLogger, WorkerResolver
#   src/utils/seed_all.py         -- Seeder
#   src/models/yolo_registry.py   -- YoloRegistry, DEFAULT_MODELS
#   src/models/pseudo_box.py      -- PseudoBoxGenerator (Otsu / locmap)
#   src/models/medgemma.py        -- MedGemmaService, MedGemmaAPI
#   src/training/yolo_dataset.py  -- load_splits, YoloDatasetBuilder
#   src/training/train_yolo.py    -- YoloTrainer (stable hyperparams)
#   src/training/yolo_callbacks.py-- epoch / NaN-guard callbacks
#   src/xai/overlays.py           -- OverlaySaver
#   evaluation/evaluate_model.py  -- YoloEvaluator
#   evaluation/yolo_plots.py      -- YoloPlotter
# =============================================================================

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List

import torch

# ---------------------------------------------------------------------------
# NMS patch -- must execute before any ultralytics import
# ---------------------------------------------------------------------------
from src.utils.nms_patch import apply_runtime_nms_patch, patch_ultralytics_nms_on_disk

patch_ultralytics_nms_on_disk()
apply_runtime_nms_patch()

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from src.models.medgemma import MedGemmaAPI, MedGemmaService
from src.models.pseudo_box import PseudoBoxGenerator
from src.models.yolo_registry import DEFAULT_MODELS, YoloRegistry
from src.training.train_yolo import YoloTrainer
from src.training.yolo_dataset import YoloDatasetBuilder, load_splits
from src.utils.io_ops import LiveLogger, WorkerResolver
from src.utils.seed_all import Seeder
from src.xai.overlays import OverlaySaver
from evaluation.evaluate_model import YoloEvaluator
from evaluation.yolo_plots import YoloPlotter


# ===========================================================================
# INLINE HELPERS  (used only in this orchestration script)
# ===========================================================================

def _get_device_list(single_gpu: bool) -> List[int]:
    """Return list of CUDA device indices, or empty list for CPU."""
    n = torch.cuda.device_count()
    if n == 0:
        return []
    return [0] if single_gpu else list(range(n))


def _free_gb(path: Path) -> float:
    """Disk space free on the partition containing ``path``, in GB."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:
        return 999.0


def _purge_corrupt_checkpoints(out_dir: Path, log: LiveLogger) -> None:
    """
    Delete any ``last.pt`` files whose weights contain NaN / Inf.
    Prevents a previous crash from poisoning the next training run.
    """
    purged = 0
    for pt in out_dir.rglob("last.pt"):
        try:
            ck    = torch.load(str(pt), map_location="cpu")
            state = ck.get("model", ck.get("state", {}))
            if hasattr(state, "state_dict"):
                state = state.state_dict()
            corrupt = False
            if isinstance(state, dict):
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            corrupt = True
                            break
            if corrupt:
                log.log("[purge] Deleting corrupt checkpoint: %s" % pt)
                pt.unlink()
                purged += 1
            else:
                log.log("[purge] Checkpoint clean: %s" % pt)
        except Exception as e:
            log.log("[purge] Cannot load %s (%s) -- deleting" % (pt, e))
            try:
                pt.unlink()
                purged += 1
            except Exception:
                pass
    log.log("[purge] Done. Removed %d corrupt checkpoint(s)." % purged)


# ===========================================================================
# ORCHESTRATION
# ===========================================================================

def _run_api_server(args) -> None:
    """Start the MedGemma HTTP API server (blocking)."""
    service = MedGemmaService(
        model_id         = args.medgemma_id,
        cache_dir        = args.medgemma_cache,
        use_4bit         = True,
        local_files_only = True,
    )
    api = MedGemmaAPI(service, host=args.medgemma_host, port=args.medgemma_port)
    print("[main] Starting MedGemma API on %s:%d" % (args.medgemma_host, args.medgemma_port))
    api.serve()


def _run_training(args, log: LiveLogger) -> None:
    """Full YOLO training pipeline: dataset build -> train -> eval -> overlays."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry    = YoloRegistry()
    variants    = [v.strip() for v in args.models.split(",")]
    device_list = _get_device_list(args.single_gpu)
    workers     = WorkerResolver.resolve(explicit=args.workers if args.workers > 0 else None)

    log.log("[main] Variants : %s" % variants)
    log.log("[main] Devices  : %s" % (device_list or "CPU"))
    log.log("[main] Workers  : %d" % workers)
    log.log("[main] Disk free: %.1f GB" % _free_gb(out_dir))

    # --- Load splits ---
    tr, va, te, names = load_splits(
        split_root = Path(args.data_root),
        data_root  = Path(args.data_root),
        train_frac = args.train_frac,
        val_frac   = args.val_frac,
        seed       = args.seed,
    )
    log.log("[main] Classes: %s" % names)
    log.log("[main] Splits  train=%d  val=%d  test=%d" % (len(tr), len(va), len(te)))

    # --- Pseudo-box generator ---
    device = torch.device("cuda:0" if device_list else "cpu")
    gen    = PseudoBoxGenerator(
        exp7_dir = args.exp7_dir,
        device   = device,
        logger   = log,
    )

    # --- Build YOLO dataset ---
    yolo_root = out_dir / "yolo_dataset"
    yaml_path = yolo_root / "dataset.yaml"
    if not args.skip_build or not yaml_path.exists():
        builder   = YoloDatasetBuilder(
            yolo_root   = yolo_root,
            gen         = gen,
            log         = log,
            copy_images = args.copy_images,
        )
        yaml_path = builder.build({"train": tr, "val": va, "test": te}, names)
    else:
        log.log("[main] Skipping dataset build (--skip_build)")

    # --- Purge corrupt checkpoints from previous crashes ---
    _purge_corrupt_checkpoints(out_dir, log)

    # --- Train and evaluate each variant ---
    all_results: Dict[str, Dict] = {}
    evaluator   = YoloEvaluator(class_names=names, log=log)
    plotter     = YoloPlotter()

    for variant in variants:
        log.log(("=" * 80))
        log.log("[main] === %s ===" % variant)

        try:
            weights_path = registry.resolve(variant, Path(args.weights_dir))
        except (ValueError, FileNotFoundError) as e:
            log.log("[main] SKIP %s: %s" % (variant, e))
            continue

        trainer = YoloTrainer(
            variant      = variant,
            weights_path = weights_path,
            yaml_path    = yaml_path,
            out_dir      = out_dir,
            epochs       = args.epochs,
            imgsz        = args.imgsz,
            batch        = args.batch,
            device_list  = device_list,
            workers      = workers,
            patience     = args.patience,
            log          = log,
        )
        try:
            best_pt = trainer.train()
        except RuntimeError as e:
            log.log("[main] %s training failed: %s" % (variant, e))
            log.log(traceback.format_exc())
            continue

        results = evaluator.evaluate_all(
            best_pt     = best_pt,
            yaml_path   = yaml_path,
            out_dir     = out_dir,
            variant     = variant,
            imgsz       = args.imgsz,
            batch       = args.batch,
            device_list = device_list,
        )
        all_results[variant] = results

        # Overlays
        saver = OverlaySaver(
            best_pt     = best_pt,
            gen         = gen,
            out_dir     = out_dir / "overlays" / variant,
            device_list = device_list,
            imgsz       = args.imgsz,
            log         = log,
        )
        saver.save(te, names, n=args.overlay_n)

    # --- Summary plots ---
    if all_results:
        plotter.save(all_results, names, out_dir / "plots", log)

    # --- Persist results JSON ---
    results_path = out_dir / "all_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.log("[main] Results saved -> %s" % results_path)


# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "exp8_localization.py",
        description = "YOLO skin lesion localisation -- Exp8",
    )

    # Data
    p.add_argument("--data_root",   default="data/isic",
                   help="Root with class subdirs or pre-split train/val/test dirs")
    p.add_argument("--train_frac",  type=float, default=0.70)
    p.add_argument("--val_frac",    type=float, default=0.15)
    p.add_argument("--copy_images", action="store_true",
                   help="Copy images instead of symlinking (needed on Windows)")

    # YOLO
    p.add_argument("--models",      default=DEFAULT_MODELS,
                   help="Comma-separated YOLO variant names, e.g. yolov11x,yolov10x")
    p.add_argument("--weights_dir", default="weights",
                   help="Directory containing pretrained .pt files")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--imgsz",       type=int,   default=640)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--patience",    type=int,   default=30)
    p.add_argument("--workers",     type=int,   default=0,
                   help="DataLoader workers (0 = auto)")
    p.add_argument("--single_gpu",  action="store_true",
                   help="Force single-GPU mode (bypass DDP)")
    p.add_argument("--skip_build",  action="store_true",
                   help="Reuse an existing YOLO dataset (skips pseudo-box generation)")

    # Output
    p.add_argument("--out_dir",     default="outputs/exp8",
                   help="Experiment output directory")
    p.add_argument("--overlay_n",   type=int,   default=50,
                   help="Number of overlay visualisation images per model")

    # Exp7 checkpoint (for locmap pseudo-box mode)
    p.add_argument("--exp7_dir",    default=None,
                   help="Path to Exp7 output dir containing best_model.pth")

    # Reproducibility
    p.add_argument("--seed",        type=int,   default=42)

    # MedGemma API server
    p.add_argument("--serve_medgemma",  action="store_true",
                   help="Start the MedGemma clinical report API server")
    p.add_argument("--medgemma_id",     default="google/medgemma-4b-it",
                   help="HuggingFace model ID for MedGemma")
    p.add_argument("--medgemma_cache",  default="hf_cache",
                   help="Local directory for HuggingFace model cache")
    p.add_argument("--medgemma_host",   default="0.0.0.0")
    p.add_argument("--medgemma_port",   type=int, default=8787)

    return p


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    args = _make_parser().parse_args()

    Seeder(args.seed).seed_everything()

    if args.serve_medgemma:
        _run_api_server(args)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = LiveLogger(out_dir / "exp8_log.txt")

    try:
        _run_training(args, log)
    except Exception as e:
        log.log("[main] FATAL: %s" % e)
        log.log(traceback.format_exc())
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
