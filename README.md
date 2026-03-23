# TRACE
## Transformative Research in Automated Clinical Evaluation
### Skin Lesion Clinical Decision Support System

**FAST-NUCES Final Year Project**

TRACE is a deployed clinical decision support system for dermoscopy-based skin lesion
assessment. It combines a ConvNeXt-Base classifier, SOTA YOLO detectors, GradCAM++
explainability, and MedGemma-4B-IT structured clinical reports into a single pipeline.

---

## System Overview

```
Dermoscopy Image
      │
      ├──▶ ConvNeXt-Base Classifier  ──▶  Class prediction + confidence
      │         │
      │         └──▶ GradCAM++  ──▶  Attention map + spatial statistics
      │                   │
      │                   └──▶ XAIReporter  ──▶  Structured XAI report
      │
      ├──▶ YOLO Localizer  ──▶  Bounding box + class
      │         │
      │         └──▶ OverlaySaver  ──▶  GT vs prediction comparison
      │
      └──▶ MedGemma-4B-IT  ──▶  Free-text clinical report
                │
                └──▶ ClinicalReporter  ──▶  Rule-based fallback (always available)
```

---

## Target Classes

| Class | Full Name | ICD-10 | Risk | Action |
|---|---|---|---|---|
| `BCC` | Basal Cell Carcinoma | C44 | High | Refer within 2 weeks |
| `BKL` | Benign Keratosis-like Lesion | L82 | Low | Annual surveillance |
| `MEL` | Melanoma | C43 | Critical | Urgent same-day referral |
| `NV` | Melanocytic Nevi | D22 | Low | Routine surveillance |

---

## Project Structure

```
TRACE_v2/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── convnext_base.yaml       ConvNeXt training hyperparameters
│   └── yolo_exp8.yaml           YOLO Exp8 training hyperparameters
│
├── experiments/
│   ├── train_convnext.py        ConvNeXt pipeline entry point
│   └── train_yolo_exp8.py       YOLO + MedGemma pipeline entry point
│
├── audit/                       Data integrity verification (4 methods)
│   ├── audit_logger.py
│   ├── method1_exact_hash.py    MD5 byte-perfect duplicate detection
│   ├── method2_phash.py         pHash near-duplicate detection
│   ├── method3_embedding.py     ConvNeXt embedding + DBSCAN clustering
│   ├── method4_hard_crop.py     Hard crop / greyscale shortcut probe
│   └── audit_runner.py          Orchestrates all 4 methods → audit_report.json
│
├── src/
│   ├── models/
│   │   ├── classifier.py        ConvNeXtClassifier (nn.Module)
│   │   ├── localizer.py         ModelLocalizer — 7 weight formats
│   │   ├── registry.py          ModelRegistry
│   │   ├── yolo_registry.py     YoloRegistry — 7 YOLO variants
│   │   ├── pseudo_box.py        PseudoBoxGenerator (Otsu + locmap)
│   │   └── medgemma.py          MedGemmaService + MedGemmaAPI (REST)
│   │
│   ├── preprocessing/
│   │   ├── __init__.py          ResizePad, TransformBuilder,
│   │   │                        SkinDatasetLoader, ClassWeightComputer,
│   │   │                        DataLoaderFactory
│   │   └── leakage.py           LeakageChecker (hard-fail on path overlap)
│   │
│   ├── training/
│   │   ├── callbacks.py         EarlyStopping, ModelCheckpoint
│   │   ├── train_classifier.py  ClassifierTrainer.fit() → history dict
│   │   ├── train_localizer.py   WarmupCosine LR scheduler
│   │   ├── train_yolo.py        YoloTrainer + STABLE_TRAIN_DEFAULTS
│   │   ├── yolo_callbacks.py    Epoch logging + NaN guard callbacks
│   │   └── yolo_dataset.py      YoloDatasetBuilder, load_splits
│   │
│   ├── xai/
│   │   ├── gradcam.py           GradCAMPlusPlus, XAIAnalyser
│   │   ├── overlays.py          OverlaySaver — YOLO GT vs prediction
│   │   └── xai_reporter.py      XAIReporter + domain knowledge tables
│   │
│   ├── utils/
│   │   ├── config_loader.py     TrainingConfig (dataclass), ConfigLoader
│   │   ├── io_ops.py            LiveLogger, JsonIO, WorkerResolver
│   │   ├── nms_patch.py         DDP NMS CPU fallback patch
│   │   └── seed_all.py          Seeder.seed_everything()
│   │
│   └── inference/
│       └── __init__.py          InferencePipeline — deployment entry point
│
├── evaluation/
│   ├── evaluate_model.py        ConvNextEvaluator, YoloEvaluator
│   ├── metrics.py               MetricsCalculator, YoloMetricsExtractor
│   │                            + 8 pure metric functions
│   ├── plotter.py               Plotter — training curves, CM, ROC, GradCAM
│   ├── reporter.py              ClinicalReporter (rule-based)
│   └── xai_gradcam.py           Standalone GradCAM++ script (post-crash use)
│
├── docs/
│   ├── classification.md        Full ConvNeXt pipeline technical docs
│   ├── localization.md          Full YOLO pipeline technical docs
│   └── xai.md                   Full XAI + clinical reports technical docs
│
├── model_cards/
│   ├── convnext_base.md         ConvNeXt model card
│   └── yolo_exp8.md             YOLO Exp8 model card
│
├── monitoring/
│   └── performance_tracker.py   PredictionLogger, DriftDetector,
│                                PerformanceSummary
│
├── scripts/
│   ├── run_convnext.sh          SLURM — ConvNeXt training
│   ├── run_yolo_exp8.sh         SLURM — YOLO training
│   ├── run_audit.sh             SLURM — data integrity audit
│   └── run_xai.sh               SLURM — standalone XAI
│
└── tests/
    ├── unit/                    12 test files — no GPU required
    │   ├── test_audit_logger.py
    │   ├── test_audit_method1.py
    │   ├── test_audit_method4.py
    │   ├── test_callbacks.py
    │   ├── test_leakage.py
    │   ├── test_model.py
    │   ├── test_preprocessing.py
    │   ├── test_scheduler.py
    │   ├── test_training.py
    │   ├── test_utils.py
    │   ├── test_xai.py
    │   └── test_xai_reporter.py
    └── integration/             3 test files — synthetic data, no GPU
        ├── test_audit_pipeline.py
        ├── test_convnext_pipeline.py
        └── test_yolo_pipeline.py
```

---

## Quick Start

### 1. Install dependencies

```bash
# Install PyTorch with CUDA first (https://pytorch.org/get-started)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install everything else
pip install -r requirements.txt
```

### 2. Run data integrity audit

Always run this before training to verify the dataset is clean.

```bash
python -c "
from audit.audit_runner import AuditRunner
from pathlib import Path
AuditRunner(
    data_root='./dataset_split',
    audit_out=Path('./audit_results'),
    checkpoint_path='./convNext/best_convnext_checkpoint.pth',  # optional
).run()
"
```

### 3. Train ConvNeXt classifier

```bash
python experiments/train_convnext.py \
    --config     configs/convnext_base.yaml \
    --data_root  ./dataset_split \
    --output_dir ./convNext \
    --epochs     60
```

### 4. Train YOLO localizer

```bash
python experiments/train_yolo_exp8.py \
    --data_root   ./dataset_split \
    --output_dir  ./ExperiemntNo8 \
    --weights_dir ./weights \
    --models      yolov11x,yolov10x,yolov9e \
    --epochs      100
```

### 5. Standalone XAI (if training crashed after saving checkpoint)

```bash
python evaluation/xai_gradcam.py \
    --checkpoint ./convNext/best_convnext_checkpoint.pth \
    --data_root  ./dataset_split \
    --split      test \
    --num_samples 40
```

### 6. Run all tests

```bash
pytest tests/unit/ -v          # fast, no GPU
pytest tests/integration/ -v   # synthetic data, still no GPU
```

### 7. Start MedGemma API server

```bash
export HF_TOKEN=hf_your_token_here

python experiments/train_yolo_exp8.py \
    --serve_medgemma \
    --medgemma_port 8787

# Send a report request
curl -X POST http://localhost:8787/report \
  -F "image=@lesion.jpg" \
  -F "pred_class=MEL" \
  -F "pred_conf=0.87" \
  -F "box_cx=0.52" -F "box_cy=0.48" \
  -F "box_w=0.31"  -F "box_h=0.28"
```

### 8. Single-image inference (deployment)

```python
from src.inference import InferencePipeline

pipe   = InferencePipeline("./convNext/best_convnext_checkpoint.pth")
result = pipe.predict_image("patient_lesion.jpg")

print(result["prediction"])   # e.g. "MEL"
print(result["confidence"])   # e.g. 0.873
print(result["class_probs"])  # {"BCC": 0.03, "BKL": 0.05, "MEL": 0.87, "NV": 0.05}
```

---

## SLURM (HPC)

```bash
mkdir -p logs

# Recommended run order
sbatch scripts/run_audit.sh        # 1. Verify dataset
sbatch scripts/run_convnext.sh     # 2. Classification
sbatch scripts/run_yolo_exp8.sh    # 3. Localisation
sbatch scripts/run_xai.sh          # 4. XAI (if needed separately)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `ResizePad` — not `Resize` | Preserves aspect ratio; black fill matches dermoscopy hardware vignette |
| Progressive fine-tuning (2 phases) | Backbone frozen during warmup prevents catastrophic forgetting of ImageNet features |
| 3-tier differential LR | Head: 2e-4 / Stage-7: 2e-5 / Rest: 5e-6 — proportional to distance from head |
| `warmup_bias_lr=0.01` | Default 0.1 caused `cls_loss=157.5` NaN explosion at epoch 4 in Exp8 run 1 |
| `amp=False` for YOLO | fp16 amplifies NaN propagation with noisy pseudo-labels |
| Pseudo-box fallback `(0.5, 0.5, 0.40, 0.40)` | Default 0.75 box covered 56% image area and caused NaN cascade |
| GradCAM++ over GradCAM | Better focal activation for small dermoscopy lesion features (2nd/3rd-order weights) |
| Target layer `features[7][2].block[0]` | Last depthwise 7×7 conv before GAP — 16×16 spatial, 1024 channels, highest semantics |
| 4-method data integrity audit | Each method catches a different class of leakage; defence-in-depth approach |
| Deployment framing | System augments clinician judgement — not a standalone diagnostic tool |

---

## Evaluation Metrics

### Classification (ConvNeXt)

| Metric | Monitor | Description |
|---|---|---|
| `macro_f1` | ✅ Primary | Equal weight per class |
| `balanced_acc` | | Mean per-class recall |
| `mcc` | | Matthews Correlation Coefficient |
| `kappa` | | Cohen's κ — agreement beyond chance |
| `macro_auc_ovr` | | Macro AUC one-vs-rest |
| `macro_pr_auc` | | Mean PR-AUC per class |

### Localisation (YOLO)

| Metric | Description |
|---|---|
| `mAP_50` | mAP at IoU threshold 0.5 |
| `mAP_50_95` | mAP averaged over IoU 0.5:0.05:0.95 |
| `precision` | Mean precision across classes |
| `recall` | Mean recall across classes |
| `per_class_AP` | Per-class AP@0.5 dict |

---

## Model Outputs

### ConvNeXt saves 7 weight formats

| File | Format |
|---|---|
| `best_convnext_weights.pth` | PyTorch state dict |
| `best_convnext_full_model.pth` | Full model object |
| `best_convnext_weights.h5` | HDF5 gzip-compressed |
| `best_convnext_weights.pkl` | Python pickle |
| `best_convnext_weights.joblib` | joblib compress=3 |
| `best_convnext_quantised_qint8.pt` | Dynamic int-8 quantised |
| `best_convnext.onnx` | ONNX opset-17, dynamic batch |

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/classification.md`](docs/classification.md) | Architecture, ResizePad, progressive training, augmentation, WarmupCosine math, all metrics |
| [`docs/localization.md`](docs/localization.md) | YOLO variants, Otsu algorithm, NaN guard, STABLE_TRAIN_DEFAULTS, MedGemma API |
| [`docs/xai.md`](docs/xai.md) | GradCAM++ derivation, target layer selection, XAIReporter, MedGemma prompt design |
| [`model_cards/convnext_base.md`](model_cards/convnext_base.md) | Intended use, training data, limitations, how to load |
| [`model_cards/yolo_exp8.md`](model_cards/yolo_exp8.md) | Pseudo-label strategy, stability fixes, evaluation table |

---

## Core Dependencies

```
torch >= 2.1.0            torchvision >= 0.16.0
ultralytics >= 8.0.0      scikit-learn >= 1.3.0
scipy >= 1.11.0           numpy >= 1.24.0
transformers >= 4.40.0    accelerate >= 0.27.0
bitsandbytes >= 0.41.0    imagededup >= 0.3.2
umap-learn >= 0.5.4       flask >= 3.0.0
pytest >= 7.4.0
```

Full list: [`requirements.txt`](requirements.txt)

---

## Clinical Decision Support Notice

TRACE is a deployed clinical decision support system. All AI-generated predictions,
attention maps, bounding boxes, and clinical reports are designed to assist qualified
dermatologists — they augment clinical judgement and do not replace it. Final diagnosis
and treatment decisions remain the responsibility of the treating clinician.