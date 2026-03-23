# TRACE — Skin Lesion Classification
## ConvNeXt-Base Progressive Fine-Tuning Pipeline

**TRACE** — *Transformative Research in Automated Clinical Evaluation*
Final Year Project · FAST-NUCES

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Dataset & Preprocessing](#dataset--preprocessing)
4. [Data Integrity Audit](#data-integrity-audit)
5. [Training Strategy](#training-strategy)
6. [Augmentation Pipeline](#augmentation-pipeline)
7. [Loss Function & Class Weights](#loss-function--class-weights)
8. [Optimiser & Learning Rate Schedule](#optimiser--learning-rate-schedule)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Model Artefacts](#model-artefacts)
11. [API Endpoint](#api-endpoint)
12. [Configuration Reference](#configuration-reference)
13. [Module Reference](#module-reference)
14. [Running the Pipeline](#running-the-pipeline)
15. [Output Structure](#output-structure)

---

## Overview

This pipeline trains a **ConvNeXt-Base** backbone for 4-class dermoscopy image classification:

| Class | Full Name | ICD-10 | Risk |
|---|---|---|---|
| `BCC` | Basal Cell Carcinoma | C44 | High |
| `BKL` | Benign Keratosis-like Lesion | L82 | Low |
| `MEL` | Melanoma | C43 | Critical |
| `NV` | Melanocytic Nevi | D22 | Low |

Key design decisions over a standard fine-tuning baseline:

- **No geometric distortion** — `ResizePad` preserves aspect ratio via longest-edge resize + symmetric zero-padding
- **Progressive fine-tuning** — backbone frozen for warmup epochs, then differential learning rates unlock stage-by-stage
- **Inline leakage guard** — hard-fails before GPU starts if any file path exists in two splits
- **Honest train metrics** — a separate eval-transform loader (no augmentation) reports unbiased training accuracy
- **Drop-last safe denominator** — loss normalised by `inputs.size(0)` not dataset length, so partial last batches do not inflate loss

---

## Architecture

### Backbone

**ConvNeXt-Base** pretrained on ImageNet-1K (`ConvNeXt_Base_Weights.IMAGENET1K_V1`).

| Property | Value |
|---|---|
| Input resolution | 512 × 512 (configurable) |
| Backbone output dim | 1024 |
| Pretrained weights | ImageNet-1K |
| Stage count | 4 (indices 0–3, feature stages 0–7) |
| Depthwise conv kernel | 7 × 7 |

### Classification Head

The original `model.classifier[2]` linear layer is replaced with a two-layer MLP:

```
Dropout(p=0.4)
→ Linear(1024, 512)
→ GELU()
→ Dropout(p=0.2)
→ Linear(512, num_classes)
```

This is implemented in `src/models/classifier.py` → `ConvNeXtClassifier`.

### GradCAM++ Target Layer

```
model._backbone.features[7][2].block[0]
```

This is the **last depthwise 7×7 convolution in stage-7, block-2** — the deepest spatially-resolved feature map before global average pooling. It captures the most semantically rich spatial information for saliency mapping.

---

## Dataset & Preprocessing

### Expected Directory Layout

```
<data_root>/
    train/
        BCC/   *.jpg  *.png  …
        BKL/   *.jpg  *.png  …
        MEL/   *.jpg  *.png  …
        NV/    *.jpg  *.png  …
    validation/          ← also accepts "val/"
        BCC/ BKL/ MEL/ NV/
    test/
        BCC/ BKL/ MEL/ NV/
```

### ResizePad Transform

Standard `transforms.Resize` squashes non-square images. `ResizePad` avoids this:

```
1. Scale longest edge → target_size (bilinear interpolation)
2. Compute symmetric padding on shorter edge
3. Pad with (0, 0, 0) black — standard for medical imaging
```

**Why black padding?** Dermoscopy images commonly have dark vignette borders; black padding is indistinguishable from natural image edges, preventing the model from learning padding artefacts.

```python
class ResizePad:
    def __init__(self, target_size: int, fill_color=(0, 0, 0))
    def __call__(self, img: Image.Image) -> Image.Image
```

### Class Weight Computation

Inverse-frequency weighting to counteract class imbalance:

```
weight_i = N_total / (N_classes × N_i)
```

Implemented inline in `DatasetLoader.load()`.
These weights are passed directly to `nn.CrossEntropyLoss(weight=...)`.

---

## Data Integrity Audit

Before training, four independent leakage checks run via `audit/`:

### Method 1 — MD5 Exact Hash (`ExactHashChecker`)

Computes MD5 checksums for every image. Any hash appearing in more than one split is flagged as a **byte-perfect duplicate** — definitive data leakage. Hard-fails training if found.

### Method 2 — pHash Near-Duplicate (`PHashChecker`)

Perceptual hashing via `imagededup` + Union-Find clustering at Hamming distance ≤ 10. Catches the same ISIC lesion photographed multiple times across splits where MD5 misses due to minor resize/compression differences.

### Method 3 — Embedding DBSCAN (`EmbeddingChecker`)

Extracts 1024-dim L2-normalised ConvNeXt-Base embeddings for every image. DBSCAN (ε=0.15, min_samples=2, Euclidean on unit sphere ≈ cosine distance) clusters visually similar images. Cross-split clusters indicate same-patient images with different framing/lighting. Optionally saves a 2D UMAP scatter plot.

### Method 4 — Hard Crop Probe (`HardCropProbe`)

Evaluates an existing checkpoint under five hostile transforms:

| Condition | Transform | What it strips |
|---|---|---|
| A. Standard | CenterCrop 512 | — baseline — |
| B. Hard Crop 70% | CenterCrop 358 → resize 512 | Background context |
| C. Greyscale | 3-channel grey | Skin-tone / colour shortcuts |
| D. Heavy Aug | Rotation 90° + ColorJitter | Spatial layout shortcuts |
| E. Combined | All of the above | Everything non-lesion |

A drop > 10% accuracy on conditions B or C indicates the model learned shortcuts rather than lesion morphology.

### Inline Leakage Check (`LeakageChecker`)

Even when the full audit is skipped, `load_datasets()` runs an inline check via `src/preprocessing/leakage.py` → `LeakageChecker`:

```python
checker = LeakageChecker(log)
checker.check(train_ds, val_ds, test_ds)   # sys.exit(1) on any overlap
```

Also verifies `class_to_idx` is alphabetically identical across all splits.

---

## Training Strategy

### Progressive Fine-Tuning

Training runs in two phases:

**Phase 1 — Warmup (epochs 1 → warmup_epochs)**

- Backbone fully frozen: only classification head gradients flow
- Single learning rate: `lr_head = 2e-4`
- Warm-up ramp: LR increases linearly from `0 → lr_head` over `warmup_epochs`

**Phase 2 — Full Fine-Tuning (epoch warmup_epochs+1 → end)**

Backbone unfrozen with three differential learning rate tiers:

| Parameter Group | Default LR | Rationale |
|---|---|---|
| Classification head | `2e-4` | New weights, high plasticity |
| Stage-7 (last feature stage) | `2e-5` | Adjacent to head, moderate update |
| Remaining backbone (stages 0–6) | `5e-6` | Pre-trained features, minimal disturbance |

This is implemented in `ConvNeXtClassifier.param_groups()` which explicitly builds three `AdamW` parameter groups with no overlap (verified by id-set intersection).

### Early Stopping

`EarlyStopping(monitor="val_macro_f1", patience=12, mode="max")` — stops when macro-F1 on the validation set does not improve by `≥ 1e-5` for 12 consecutive epochs.

`ModelCheckpoint` keeps an in-memory copy of the best weights and saves a `.pth` checkpoint dict containing:
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "class_names": List[str],
    "best_val_macro_f1": float,
    "best_epoch": int,
    "args": dict,
}
```

---

## Augmentation Pipeline

### Training Transforms (in order)

| Transform | Parameters | Purpose |
|---|---|---|
| `ResizePad` | target=512, fill=(0,0,0) | Aspect-ratio-safe resize |
| `RandomHorizontalFlip` | p=0.5 | Left-right symmetry |
| `RandomVerticalFlip` | p=0.3 | Dermoscopy images have no fixed orientation |
| `RandomRotation` | degrees=25 | Rotation invariance |
| `ColorJitter` | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Lighting variation |
| `RandomAffine` | translate=(0.08,0.08), scale=(0.90,1.10) | Minor position / scale shift |
| `RandomGrayscale` | p=0.05 | Colour channel robustness |
| `GaussianBlur` | kernel=3, σ=(0.1,1.5) | Focus variation |
| `ToTensor` | — | Convert to float32 in [0,1] |
| `Normalize` | μ=(0.485,0.456,0.406), σ=(0.229,0.224,0.225) | ImageNet normalisation |
| `RandomErasing` | p=0.15, scale=(0.02,0.08) | Occlusion robustness |

### Eval Transforms (deterministic)

`ResizePad(512)` → `ToTensor` → `Normalize` (same μ/σ)

### Mixup Augmentation

Applied with probability 0.5 per batch:

```
λ ~ Beta(α, α),  α = 0.3
x_mixed = λ·x_i + (1-λ)·x_j
L_mixed = λ·CE(y_i) + (1-λ)·CE(y_j)
```

Mixup smooths the decision boundary and acts as an additional regulariser alongside label smoothing.

---

## Loss Function & Class Weights

**Weighted Cross-Entropy with Label Smoothing:**

```
L = -Σ w_c · [ (1-ε)·y_c·log(p_c) + ε/K·log(p_c) ]
```

where:
- `w_c` = inverse-frequency class weight
- `ε` = 0.1 (label smoothing)
- `K` = 4 (number of classes)

Label smoothing prevents overconfident predictions and improves calibration on the imbalanced dermoscopy dataset.

---

## Optimiser & Learning Rate Schedule

### AdamW

```python
optim.AdamW(param_groups, weight_decay=1e-4)
```

`weight_decay=1e-4` applies proper L2 regularisation decoupled from the gradient update (unlike the original Adam weight decay). No L1 hack needed.

### WarmupCosine Scheduler

Two-phase schedule:

**Warm-up phase** (epochs 0 → warmup_epochs − 1):
```
LR(e) = base_lr × (e+1) / warmup_epochs
```

**Cosine annealing phase** (epochs warmup_epochs → end):
```
LR(e) = η_min + (base_lr − η_min) × 0.5 × (1 + cos(π·t/T))
```
where `t = epoch − warmup_epochs`, `T = total_epochs − warmup_epochs`, `η_min = 1e-7`.

The scheduler is re-instantiated with `warmup=0` after backbone unfreeze so the full cosine budget applies to the fine-tuning phase only.

### Mixed Precision (AMP)

```python
torch.amp.GradScaler("cuda", enabled=amp_on)
torch.amp.autocast("cuda", enabled=amp_on)
```

Using the updated torch 2.x AMP API (no deprecation warnings). Gradient clipping is applied **after** `scaler.unscale_()`:

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## Evaluation Metrics

All metrics computed in `evaluation/metrics.py` via `MetricsCalculator`:

| Metric | Description |
|---|---|
| `acc` | Standard accuracy |
| `balanced_acc` | Mean per-class recall — robust to imbalance |
| `mcc` | Matthews Correlation Coefficient ∈ [-1, 1] |
| `kappa` | Cohen's κ — agreement beyond chance |
| `macro_f1` | Macro-averaged F1 — equal class weight (primary monitor) |
| `weighted_f1` | Support-weighted F1 |
| `macro_auc_ovr` | Macro AUC one-vs-rest |
| `macro_auc_ovo` | Macro AUC one-vs-one |
| `macro_pr_auc` | Mean area under Precision-Recall curve |

Evaluation runs on **three splits** (train with eval-transforms, validation, test) and produces:
- Per-split classification reports (4-decimal digits)
- Confusion matrices (counts + normalised)
- ROC curves per class
- Precision-Recall curves per class
- Per-class P/R/F1 bar charts
- Confidence distribution histograms

---

## Model Artefacts

The primary production checkpoint is saved by `save_artifacts()` in `trace_convnext.py`:

**Production checkpoint:** `weights/convnext/best_convnext_checkpoint.pth`

This file contains the full checkpoint dict:
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "class_names": List[str],
    "best_val_macro_f1": float,
    "best_epoch": int,
    "args": dict,
}
```

Additional export formats are also saved:

| File | Format | Use Case |
|---|---|---|
| `best_convnext_checkpoint.pth` | Full checkpoint dict | Resume training / production API |
| `best_convnext_weights.pth` | PyTorch state dict | Standard PyTorch reload |
| `best_convnext_full_model.pth` | Full model object | Quick inference without rebuilding arch |
| `best_convnext_weights.h5` | HDF5 gzip-compressed | Cross-framework / archival |
| `best_convnext_weights.pkl` | Python pickle (numpy) | NumPy-only environments |
| `best_convnext_weights.joblib` | joblib compress=3 | Sklearn-compatible environments |
| `best_convnext_quantised_qint8.pt` | Dynamic int-8 quantised | CPU deployment |
| `best_convnext.onnx` | ONNX opset-17, dynamic batch | Production inference / TensorRT |

DataParallel wrapper is stripped before saving so all formats load on any hardware configuration.

---

## API Endpoint

The trained checkpoint is served via the TRACE FastAPI service.

### POST /classify

Classifies a single dermoscopy image using the ConvNeXt-Base checkpoint.

**Weights loaded from:** `weights/convnext/best_convnext_checkpoint.pth`

**Request:** `multipart/form-data`

| Field   | Type | Required | Description              |
|---------|------|----------|--------------------------|
| `image` | file | Yes      | JPEG / PNG / BMP / TIFF  |

**Response — 200 OK:**

```json
{
  "request_id": "uuid4",
  "model": "ConvNeXt-Base",
  "image_width": 224,
  "image_height": 224,
  "predicted_class": "MEL",
  "predicted_class_id": 2,
  "confidence": 0.9005,
  "probabilities": [
    {"class_name": "BCC", "class_id": 0, "probability": 0.0258},
    {"class_name": "BKL", "class_id": 1, "probability": 0.0593},
    {"class_name": "MEL", "class_id": 2, "probability": 0.9005},
    {"class_name": "NV",  "class_id": 3, "probability": 0.0143}
  ],
  "class_info": {
    "full_name": "Melanoma",
    "risk": "Critical",
    "icd10": "C43.9",
    "action": "URGENT: Immediate oncology referral"
  },
  "inference_time_ms": 556.1,
  "warning": "AI output is for research assistance only. Not a substitute for clinical diagnosis."
}
```

### POST /classify/batch

Upload up to 32 images; each is classified independently.
Returns `{"total_images": N, "results": [...]}` where each element is a full `/classify` response.

### Start the API server

```bash
python main_api.py --reload       # development
python main_api.py --workers 4    # production
```

Interactive docs at `http://localhost:8000/docs`.

---

## Configuration Reference

All hyperparameters live in `configs/convnext_base.yaml` and the `TrainingConfig` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `data_root` | `/home/f223085/.../dataset_split` | Root of pre-split dataset |
| `output_dir` | `./convNext` | All outputs written here |
| `image_size` | `512` | Spatial resolution after ResizePad |
| `batch_size` | `8` | Per-GPU batch size |
| `epochs` | `60` | Maximum training epochs |
| `warmup_epochs` | `5` | Backbone-frozen warm-up epochs |
| `patience` | `12` | Early stopping patience (val macro-F1) |
| `lr_head` | `2e-4` | Head learning rate |
| `lr_stage7` | `2e-5` | Stage-7 learning rate |
| `lr_rest` | `5e-6` | Remaining backbone LR |
| `weight_decay` | `1e-4` | AdamW weight decay (L2) |
| `label_smoothing` | `0.1` | Cross-entropy label smoothing ε |
| `dropout` | `0.4` | Head dropout probability |
| `mixup_alpha` | `0.3` | Beta(α,α) Mixup concentration; 0 = off |
| `grad_clip` | `1.0` | Max gradient norm (L2) |
| `gradcam_samples` | `40` | Test images visualised with GradCAM++ |
| `seed` | `42` | Master random seed |

---

## Module Reference

```
src/
├── models/
│   └── classifier.py      ConvNeXtClassifier(nn.Module)
│                            .freeze_backbone()
│                            .unfreeze_backbone()
│                            .param_groups(lr_head, lr_stage7, lr_rest)
│                            .gradcam_target_layer  [property]
│                            .count_trainable()
│                          unwrap_model(model)
│
├── preprocessing/
│   ├── transforms.py      ResizePad(target_size, fill_color)
│   │                      TransformBuilder.build_train(image_size)
│   │                      TransformBuilder.build_eval(image_size)
│   ├── dataset_loader.py  DatasetLoader(data_root, image_size, batch_size, ...)
│   │                        .load() → (train_ds, val_ds, test_ds, class_names, weights)
│   │                        .build_loaders() → (train_loader, val_loader, test_loader)
│   │                      [class weights computed inline in DatasetLoader.load()]
│   └── leakage.py         LeakageChecker.check(train, val, test)
│
├── training/
│   ├── callbacks.py       Callback (ABC)
│   │                        EarlyStopping(monitor, patience, mode)
│   │                        ModelCheckpoint(out_dir, monitor)
│   │                          .restore_best_weights(model)
│   │                        WarmupCosine(optimizer, warmup, total)
│   ├── train_classifier.py  ConvNeXtTrainer.fit() → history dict
│   └── mixup.py           MixupAugmentation(alpha)
│                            .apply(x, y) → (x_mixed, y_a, y_b, lam)
│
├── utils/
│   ├── preflight.py       PreflightChecker.run(args) → bool
│   ├── config_loader.py   TrainingConfig (dataclass)
│   │                      ConfigLoader.load() → TrainingConfig
│   ├── io_ops.py          LiveLogger, JsonIO, WorkerResolver
│   └── seed_all.py        Seeder(seed).seed_everything()
│
└── xai/
    ├── gradcam.py         GradCAMPlusPlus(model, target_layer)
    │                      GradCAMSaver(out_dir, class_names)
    │                      get_gradcam_target_layer(model) → nn.Module
    ├── visualization.py   TrainingPlotter.plot_history(history, out_dir)
    └── xai_reporter.py    XAIReporter.generate(sample) → dict

evaluation/
├── evaluate_model.py      ConvNextEvaluator.evaluate(model, loader, ...)
│                          ConvNextEvaluator.evaluate_all_splits(...)
└── metrics.py             MetricsCalculator.compute(y_true, y_pred, y_prob, ...)
                           compute_accuracy, compute_mcc, compute_kappa, ...

audit/
├── method1_exact_hash.py  ExactHashChecker.run(splits)
├── method2_phash.py       PHashChecker.run(splits)
├── method3_embedding.py   EmbeddingChecker.run(splits, output_dir)
├── method4_hard_crop.py   HardCropProbe.run(output_dir)
└── audit_runner.py        AuditRunner.run() → audit_report.json
```

---

## Running the Pipeline

### Full pipeline (audit then train)

```bash
python trace_convnext.py \
    --data_root /path/to/dataset_split \
    --output_dir ./convNext \
    --image_size 512 \
    --batch_size 8 \
    --epochs 60 \
    --warmup_epochs 5 \
    --lr_head 2e-4 --lr_stage7 2e-5 --lr_rest 5e-6 \
    --weight_decay 1e-4 --label_smoothing 0.1 \
    --dropout 0.4 --patience 12 \
    --mixup_alpha 0.3 --grad_clip 1.0 \
    --gradcam_samples 40 --seed 42
```

### Audit only

```bash
python -c "
from audit.audit_runner import AuditRunner
AuditRunner(
    data_root='./dataset_split',
    audit_out=__import__('pathlib').Path('./audit_results'),
    checkpoint_path='./weights/convnext/best_convnext_checkpoint.pth',
).run()
"
```

### Resume from checkpoint

```bash
python trace_convnext.py \
    --data_root /path/to/data \
    --output_dir ./convNext \
    --resume ./weights/convnext/best_convnext_checkpoint.pth
```

### Using YAML config

```bash
python trace_convnext.py --config configs/convnext_base.yaml
```

---

## Output Structure

```
convNext/
├── best_convnext_checkpoint.pth      Best checkpoint dict
├── best_convnext_weights.pth         State dict only
├── best_convnext_full_model.pth      Full model object
├── best_convnext_weights.h5          HDF5 weights
├── best_convnext_weights.pkl         Pickle weights
├── best_convnext_weights.joblib      Joblib weights
├── best_convnext_quantised_qint8.pt  Quantised model
├── best_convnext.onnx                ONNX export
├── history.json                      Per-epoch metrics (updated live)
├── metrics_summary.json              Final metrics all splits
├── training_log.txt                  Full dual-stream training log
│
├── reports/
│   ├── classification_report_train.txt
│   ├── classification_report_validation.txt
│   └── classification_report_test.txt
│
├── plots/
│   ├── 01_loss_accuracy.png
│   ├── 02_macro_f1.png
│   ├── 03_generalisation_gap.png
│   ├── 04_lr_schedule.png
│   ├── 05_balanced_accuracy.png
│   ├── {split}_cm_counts.png         (×3 splits)
│   ├── {split}_cm_normalised.png     (×3 splits)
│   ├── {split}_roc.png               (×3 splits)
│   ├── {split}_pr_curves.png         (×3 splits)
│   ├── {split}_per_class_metrics.png (×3 splits)
│   └── {split}_confidence_dist.png   (×3 splits)
│
├── gradcam/
│   ├── gradcam_NNNN_*.png            GradCAM++ overlay images
│   ├── gradcam_report.json           Per-sample GradCAM++ data
│   ├── xai_reports.json              Structured XAI reports
│   ├── xai_reports.txt               Human-readable XAI text
│   ├── clinical_reports.json         Rule-based clinical reports
│   ├── clinical_reports.txt          Human-readable clinical text
│   └── medgemma_reports.json         MedGemma reports (if enabled)
│
└── audit/
    ├── audit_log.txt
    └── audit_report.json

weights/
└── convnext/
    └── best_convnext_checkpoint.pth  ← production checkpoint
```

---

## Dependencies

```
torch >= 2.1.0
torchvision >= 0.16.0
scikit-learn >= 1.3.0
Pillow >= 10.0.0
numpy >= 1.24.0
tqdm >= 4.65.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
h5py >= 3.9.0
joblib >= 1.3.0
pyyaml >= 6.0
onnx >= 1.14.0
fastapi >= 0.110.0
uvicorn[standard] >= 0.27.0
pydantic >= 2.0.0
scipy                      # for audit Method 2
imagededup                 # optional — audit Method 2 (pip install imagededup)
umap-learn                 # optional — audit Method 3 UMAP plot
transformers, accelerate   # optional — MedGemma clinical reports
```

---

> **Clinical Decision Support:** TRACE is a deployed clinical decision support tool designed to assist qualified dermatologists. All AI-generated predictions and XAI outputs are intended to augment, not replace, clinical judgement. Final diagnosis and treatment decisions remain the responsibility of the treating clinician.
