# -*- coding: utf-8 -*-
"""
trace_convnext.py
=================
TRACE Project — FAST-NU Final Year Project
Entry point: Data Integrity Audit + ConvNeXt-Base Training Pipeline

This file is the unified CLI entry point. All logic lives in the
modular ``src/`` packages; this script wires them together and
exposes the ``--mode`` interface.

=== MODES ===
  --mode audit   Run data integrity checks BEFORE training
  --mode train   Run full ConvNeXt training pipeline
  --mode both    Run audit then train (FYP-defence-ready)

=== SOURCE MODULES ===
  src/utils/preflight.py          PreflightChecker
  src/preprocessing/transforms.py ResizePad, TransformBuilder
  src/preprocessing/dataset_loader.py DatasetLoader
  src/training/mixup.py           MixupAugmentation
  src/training/callbacks.py       WarmupCosine, EarlyStopping
  src/models/classifier.py        ConvNeXtClassifier, unwrap_model
  src/xai/gradcam.py              GradCAMPlusPlus, GradCAMSaver
  src/xai/visualization.py        TrainingPlotter
  src/training/train_classifier.py ConvNeXtTrainer
  src/audit/audit_runner.py       AuditRunner (audit pipeline)
  src/xai/xai_reporter.py         XAIReporter

=== USAGE ===
  # Full FYP-defence pipeline:
  python trace_convnext.py --mode both \\
      --data_root /data/dataset_split \\
      --output_dir ./convNext \\
      --image_size 512 --batch_size 8 --epochs 60 \\
      --warmup_epochs 5 --num_workers 12 \\
      --lr_head 2e-4 --lr_stage7 2e-5 --lr_rest 5e-6 \\
      --weight_decay 1e-4 --label_smoothing 0.1 --dropout 0.4 \\
      --patience 12 --mixup_alpha 0.3 --grad_clip 1.0 \\
      --gradcam_samples 40 --seed 42

  # Audit only:
  python trace_convnext.py --mode audit \\
      --data_root /data/dataset_split \\
      --checkpoint ./convNext/best_convnext_checkpoint.pth \\
      --output_dir ./audit_results

  # Training only (audit already passed):
  python trace_convnext.py --mode train \\
      --data_root /data/dataset_split \\
      --output_dir ./convNext --image_size 512 --batch_size 8
"""

import argparse
import copy
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

# ---------------------------------------------------------------------------
# Pre-flight check — runs before any heavy imports
# ---------------------------------------------------------------------------
from src.utils.preflight import PreflightChecker
PreflightChecker().check()

# ---------------------------------------------------------------------------
# Heavy imports (only reached if preflight passes)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Modular imports
# ---------------------------------------------------------------------------
from src.models.classifier      import ConvNeXtClassifier, unwrap_model
from src.preprocessing.dataset_loader import DatasetLoader
from src.preprocessing.transforms     import TransformBuilder
from src.training.callbacks     import WarmupCosine, EarlyStopping
from src.training.mixup         import MixupAugmentation
from src.training.train_classifier     import ConvNeXtTrainer
from src.utils.io_ops           import LiveLogger
from src.utils.seed_all         import Seeder
from src.utils.io_ops           import WorkerResolver
from src.audit.audit_runner     import AuditRunner
from src.xai.gradcam            import GradCAMSaver
from src.xai.visualization      import TrainingPlotter
from src.xai.xai_reporter       import XAIReporter


# ==============================================================================
# ARTIFACT SAVER
# Multi-format weight export: .pth state dict, full model, .h5, .pkl, .joblib,
# quantised int8, ONNX.  All saves are skip-if-exists (safe to resume).
# ==============================================================================

def save_artifacts(model, out_dir: Path, image_size: int, class_names, log: LiveLogger):
    try:
        import h5py
        h5py_available = True
    except ImportError:
        h5py_available = False
        log.log("  [warn] h5py not installed — .h5 skipped. "
                "Run: conda install -y h5py -c conda-forge")
    import pickle
    import joblib

    raw_model = unwrap_model(model)
    sd        = raw_model.state_dict()

    def _skip_or_save(path, fn, label):
        if path.exists():
            log.log(f"  [skip]  already exists: {path.name}")
        else:
            fn(path)
            log.log(f"  [saved] {label}: {path.name}")

    _skip_or_save(
        out_dir / "best_convnext_weights.pth",
        lambda p: torch.save(sd, p),
        "state dict (.pth)",
    )
    _skip_or_save(
        out_dir / "best_convnext_full_model.pth",
        lambda p: torch.save(raw_model, p),
        "full model (.pth)",
    )

    p_h5 = out_dir / "best_convnext_weights.h5"
    if p_h5.exists():
        log.log(f"  [skip]  already exists: {p_h5.name}")
    elif not h5py_available:
        log.log(f"  [skip]  h5py unavailable")
    else:
        try:
            with h5py.File(p_h5, "w") as hf:
                for key, tensor in sd.items():
                    hf.create_dataset(
                        key.replace(".", "/"),
                        data=tensor.cpu().float().numpy(),
                        compression="gzip", compression_opts=4,
                    )
                hf.attrs["image_size"]   = image_size
                hf.attrs["class_names"]  = json.dumps(class_names)
                hf.attrs["architecture"] = "convnext_base"
            log.log(f"  [saved] HDF5 weights (.h5): {p_h5.name}")
        except Exception as e:
            log.log(f"  [skip]  .h5 failed: {e}")

    payload = {k: v.cpu().numpy() for k, v in sd.items()}
    _skip_or_save(
        out_dir / "best_convnext_weights.pkl",
        lambda p: open(p, "wb").write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)) or None,
        "pickle (.pkl)",
    )
    _skip_or_save(
        out_dir / "best_convnext_weights.joblib",
        lambda p: joblib.dump(payload, p, compress=3),
        "joblib (.joblib)",
    )

    p_q = out_dir / "best_convnext_quantised_qint8.pt"
    if p_q.exists():
        log.log(f"  [skip]  already exists: {p_q.name}")
    else:
        try:
            qm = torch.quantization.quantize_dynamic(
                copy.deepcopy(raw_model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
            torch.save(qm, p_q)
            log.log(f"  [saved] quantised int8: {p_q.name}")
        except Exception as e:
            log.log(f"  [skip]  quantise failed: {e}")

    p_onnx = out_dir / "best_convnext.onnx"
    if p_onnx.exists():
        log.log(f"  [skip]  already exists: {p_onnx.name}")
    else:
        try:
            m     = copy.deepcopy(raw_model).cpu().eval()
            dummy = torch.zeros(1, 3, image_size, image_size)
            torch.onnx.export(
                m, dummy, str(p_onnx),
                input_names=["image"], output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
            )
            log.log(f"  [saved] ONNX model: {p_onnx.name}")
        except Exception as e:
            log.log(f"  [skip]  ONNX failed: {e}")


# ==============================================================================
# RUN AUDIT
# ==============================================================================

def run_audit(args, audit_out: Path):
    """Delegate to AuditRunner (src/audit/audit_runner.py)."""
    runner = AuditRunner(
        data_root           = args.data_root,
        audit_out           = audit_out,
        checkpoint_path     = getattr(args, "checkpoint", None),
        image_size          = getattr(args, "image_size", 512),
        batch_size          = getattr(args, "batch_size", 32),
        num_workers         = WorkerResolver.resolve(getattr(args, "num_workers", None)),
        phash_threshold     = getattr(args, "phash_threshold", 10),
        dbscan_eps          = getattr(args, "dbscan_eps", 0.15),
        dbscan_min_samples  = getattr(args, "dbscan_min_samples", 2),
        skip_phash          = getattr(args, "skip_phash", False),
        skip_embedding      = getattr(args, "skip_embedding", False),
        skip_hard_crop      = getattr(args, "skip_hard_crop", False),
    )
    return runner.run()


# ==============================================================================
# RUN TRAINING
# ==============================================================================

def run_training(args):
    """Full ConvNeXt-Base training pipeline using modular classes."""
    Seeder(args.seed).seed_everything()
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log       = LiveLogger(out_dir / "training_log.txt")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = WorkerResolver.resolve(args.num_workers)

    log.sep(True)
    log.log("ConvNeXt-Base Skin Cancer — TRACE Production")
    log.log(f"  device={device}  GPUs={torch.cuda.device_count()}")
    log.log(f"  image_size={args.image_size}  batch={args.batch_size}")
    log.log(f"  epochs={args.epochs}  warmup={args.warmup_epochs}  patience={args.patience}")
    log.log(f"  lr: head={args.lr_head}  stage7={args.lr_stage7}  rest={args.lr_rest}")
    log.log(f"  dropout={args.dropout}  label_smooth={args.label_smoothing}")
    log.log(f"  mixup={args.mixup_alpha}  grad_clip={args.grad_clip}  wd={args.weight_decay}")
    log.log(f"  data_root={args.data_root}")
    log.sep(True)

    # ---- Load data ----
    dataset_loader = DatasetLoader(args.data_root, args.image_size, log)
    train_ds, val_ds, test_ds = dataset_loader.load()
    class_names = train_ds.classes
    num_classes = len(class_names)

    counts       = Counter(train_ds.targets)
    total_train  = len(train_ds)
    class_weights = torch.tensor(
        [total_train / (num_classes * max(counts[i], 1)) for i in range(num_classes)],
        dtype=torch.float, device=device,
    )
    log.log(f"[data] Class counts: { {class_names[i]: counts[i] for i in range(num_classes)} }")
    log.log(f"[data] Class weights: "
            f"{ {class_names[i]: round(float(class_weights[i]), 3) for i in range(num_classes)} }")

    loader_kw = dict(
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=n_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # Eval loader on train split (no augmentation)
    _, eval_tf = TransformBuilder.build(args.image_size)
    train_eval_ds     = datasets.ImageFolder(
        str(Path(args.data_root) / "train"), transform=eval_tf)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size,
                                   shuffle=False, **loader_kw)

    # ---- Train ----
    trainer = ConvNeXtTrainer(
        num_classes     = num_classes,
        class_names     = class_names,
        out_dir         = out_dir,
        log             = log,
        device          = device,
        epochs          = args.epochs,
        warmup_epochs   = args.warmup_epochs,
        batch_size      = args.batch_size,
        lr_head         = args.lr_head,
        lr_stage7       = args.lr_stage7,
        lr_rest         = args.lr_rest,
        weight_decay    = args.weight_decay,
        label_smoothing = args.label_smoothing,
        dropout         = args.dropout,
        mixup_alpha     = args.mixup_alpha,
        grad_clip       = args.grad_clip,
        patience        = args.patience,
        gradcam_samples = args.gradcam_samples,
        num_workers     = n_workers,
    )
    history = trainer.fit(train_loader, val_loader)
    model   = trainer._model

    # ---- Final evaluation ----
    from evaluation.evaluate_model import ConvNextEvaluator
    criterion    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    evaluator    = ConvNextEvaluator(model, device, num_classes)
    split_metrics = {
        "train":      evaluator.evaluate(train_eval_loader, criterion),
        "validation": evaluator.evaluate(val_loader,        criterion),
        "test":       evaluator.evaluate(test_loader,       criterion),
    }

    for sn, sm in split_metrics.items():
        log.log(f"  {sn:12s} | acc={sm['acc']:.5f}  bal={sm['balanced_acc']:.5f}  "
                f"F1={sm['macro_f1']:.5f}  MCC={sm['mcc']:.4f}  "
                f"AUC={sm.get('macro_auc_ovr') or 0:.4f}")

    # Classification reports
    rpt_dir = out_dir / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    for sn, sm in split_metrics.items():
        rpt = classification_report(sm["labels"], sm["preds"],
                                    target_names=class_names, digits=4, zero_division=0)
        log.log(f"\n[eval] {sn} Classification Report:\n{rpt}")
        (rpt_dir / f"classification_report_{sn}.txt").write_text(rpt)

    # ---- Save artifacts ----
    log.sep()
    log.log("[save] Saving all artifacts...")
    save_artifacts(model, out_dir, args.image_size, class_names, log)

    # ---- GradCAM++ ----
    log.sep()
    gradcam_dir         = out_dir / "gradcam"
    gradcam_report_path = gradcam_dir / "gradcam_report.json"
    if gradcam_report_path.exists():
        log.log("[gradcam] Already done — loading existing report")
        gradcam_reports = json.loads(gradcam_report_path.read_text())
    else:
        log.log(f"[gradcam] Generating {args.gradcam_samples} GradCAM++ overlays on test set...")
        cam_saver = GradCAMSaver(model, class_names, gradcam_dir, device, log)
        gradcam_reports = cam_saver.run(test_loader, max_samples=args.gradcam_samples)

    # ---- XAI structured reports ----
    log.sep()
    xai_report_path = gradcam_dir / "xai_reports.json"
    if xai_report_path.exists():
        log.log("[xai] Already done — loading existing XAI reports")
        xai_reports = json.loads(xai_report_path.read_text())
    else:
        log.log(f"[xai] Generating XAI reports for {len(gradcam_reports)} samples...")
        reporter  = XAIReporter()
        xai_reports = []
        for sample in gradcam_reports:
            xai_reports.append({
                "sample_id":  sample["sample_id"],
                "pred_class": sample["pred_class"],
                "true_class": sample["true_class"],
                "confidence": sample["pred_confidence"],
                "correct":    sample["correct"],
                "xai_report": reporter.generate(sample),
            })
        xai_report_path.write_text(json.dumps(xai_reports, indent=2))
        log.log(f"[xai] {len(xai_reports)} XAI reports saved -> {xai_report_path}")

    # ---- MedGemma (optional) ----
    log.sep()
    if getattr(args, "enable_medgemma", False):
        medgemma_path = gradcam_dir / "medgemma_reports.json"
        if medgemma_path.exists():
            log.log("[medgemma] Already done — skipping")
        else:
            from src.models.medgemma import MedGemmaService
            svc = MedGemmaService(
                model_id=args.medgemma_model_id,
                max_new_tokens=args.medgemma_max_new_tokens,
            )
            svc.generate_batch(
                gradcam_reports=gradcam_reports,
                xai_reports=xai_reports,
                max_samples=args.medgemma_max_samples,
                out_dir=gradcam_dir,
                log=log,
            )
    else:
        log.log("[medgemma] Skipped (pass --enable_medgemma to activate).")

    # ---- Plots ----
    log.sep()
    plots_dir = out_dir / "plots"
    if plots_dir.exists() and any(plots_dir.iterdir()):
        log.log(f"[plots] Already done — skipping")
    else:
        log.log("[plots] Generating all plots...")
        plotter = TrainingPlotter(out_dir, log)
        plotter.save_all(history, split_metrics, class_names)

    # ---- Final summary JSON ----
    clean = lambda m: {k: v for k, v in m.items() if k not in ("preds", "labels", "probs")}
    final_gap_acc = history["train_acc"][-1] - history["val_acc"][-1]
    final_gap_f1  = history["train_macro_f1"][-1] - history["val_macro_f1"][-1]
    fit_status    = (
        "overfitting"  if final_gap_acc > 0.10 or final_gap_f1 > 0.10
        else "underfitting" if history["train_acc"][-1] < 0.70
        else "well-fit"
    )
    summary = {
        "experiment":         "ConvNeXt-Base TRACE Skin Cancer Production",
        "total_epochs_run":   len(history["train_loss"]),
        "fit_status":         fit_status,
        "generalisation_gap": {"acc": round(final_gap_acc, 4), "f1": round(final_gap_f1, 4)},
        "classes":            class_names,
        "metrics_by_split":   {sn: clean(sm) for sn, sm in split_metrics.items()},
        "history":            history,
        "args":               vars(args),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    log.sep(True)
    log.log("FINAL RESULTS")
    log.sep()
    log.log(f"  Fit status: {fit_status}")
    log.log(f"  Train-Val gap: acc={final_gap_acc:+.4f}  f1={final_gap_f1:+.4f}")
    for sn, sm in split_metrics.items():
        log.log(f"  {sn:12s} | acc={sm['acc']:.5f}  F1={sm['macro_f1']:.5f}  "
                f"AUC={sm.get('macro_auc_ovr') or 0:.4f}")
    log.sep()
    log.log(f"  Output dir: {out_dir.resolve()}")
    log.sep(True)
    log.close()


# ==============================================================================
# ARGUMENT PARSER + MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="TRACE ConvNeXt — Unified Training + Data Integrity Audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=["train", "audit", "both"], default="both",
                    help="train=training only | audit=integrity checks only | both=audit then train")

    # --- shared ---
    ap.add_argument("--data_root",   default="/home/f223085/ExperiemntNo2/dataset_split")
    ap.add_argument("--output_dir",  default="./convNext")
    ap.add_argument("--image_size",  type=int,  default=512)
    ap.add_argument("--num_workers", type=int,  default=None)
    ap.add_argument("--seed",        type=int,  default=42)

    # --- training ---
    ap.add_argument("--batch_size",              type=int,   default=8)
    ap.add_argument("--epochs",                  type=int,   default=60)
    ap.add_argument("--warmup_epochs",           type=int,   default=5)
    ap.add_argument("--lr_head",                 type=float, default=2e-4)
    ap.add_argument("--lr_stage7",               type=float, default=2e-5)
    ap.add_argument("--lr_rest",                 type=float, default=5e-6)
    ap.add_argument("--weight_decay",            type=float, default=1e-4)
    ap.add_argument("--label_smoothing",         type=float, default=0.1)
    ap.add_argument("--dropout",                 type=float, default=0.4)
    ap.add_argument("--patience",                type=int,   default=12)
    ap.add_argument("--mixup_alpha",             type=float, default=0.3)
    ap.add_argument("--grad_clip",               type=float, default=1.0)
    ap.add_argument("--gradcam_samples",         type=int,   default=40)
    ap.add_argument("--resume",                  default=None)
    ap.add_argument("--enable_medgemma",         action="store_true", default=False)
    ap.add_argument("--medgemma_model_id",       default="google/medgemma-4b-it")
    ap.add_argument("--medgemma_max_samples",    type=int,   default=15)
    ap.add_argument("--medgemma_max_new_tokens", type=int,   default=512)

    # --- audit ---
    ap.add_argument("--checkpoint",          default=None,
                    help="Checkpoint .pth for audit Methods 3 & 4")
    ap.add_argument("--audit_output_dir",    default=None,
                    help="Audit output dir (default: <output_dir>/audit)")
    ap.add_argument("--phash_threshold",     type=int,   default=10)
    ap.add_argument("--dbscan_eps",          type=float, default=0.15)
    ap.add_argument("--dbscan_min_samples",  type=int,   default=2)
    ap.add_argument("--skip_phash",          action="store_true")
    ap.add_argument("--skip_embedding",      action="store_true")
    ap.add_argument("--skip_hard_crop",      action="store_true")

    args = ap.parse_args()
    Seeder(args.seed).seed_everything()

    out_dir   = Path(args.output_dir)
    audit_dir = Path(args.audit_output_dir) if args.audit_output_dir else out_dir / "audit"

    # Auto-detect checkpoint for audit
    if args.checkpoint is None:
        candidate = out_dir / "best_convnext_checkpoint.pth"
        if candidate.exists():
            args.checkpoint = str(candidate)
            print(f"[main] Auto-detected checkpoint: {candidate}")

    if args.mode in ("audit", "both"):
        print("\n" + "=" * 78)
        print("PHASE 1: DATA INTEGRITY AUDIT")
        print("=" * 78)
        run_audit(args, audit_dir)

    if args.mode in ("train", "both"):
        print("\n" + "=" * 78)
        print("PHASE 2: CONVNEXT-BASE TRAINING PIPELINE")
        print("=" * 78)
        run_training(args)


if __name__ == "__main__":
    main()
