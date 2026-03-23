# TRACE ConvNeXt-Base Classification Pipeline

## 1. Overview

The ConvNeXt-Base classification pipeline is the primary diagnostic model in the TRACE Skin Cancer Detection system. It takes dermoscopic images as input and classifies them into one of four skin lesion categories using a pretrained ConvNeXt-Base backbone with a custom two-layer classification head.

The pipeline is a refactored, modular decomposition of the original monolithic `trace_convnext.py`. Each concern is isolated into its own module, making the code testable, maintainable, and auditable.

Key characteristics:
- ImageNet pretrained ConvNeXt-Base backbone (88M parameters)
- Progressive fine-tuning: frozen backbone during warmup, differential learning rates after
- Mixup augmentation for regularisation
- Automatic mixed precision (AMP) training with GradScaler
- Gradient clipping and label smoothing
- Early stopping monitored on validation macro-F1
- GradCAM++ explainability overlays for every test prediction
- Full clinical reporting via `XAIReporter`

---

## 2. Module Breakdown

| Class / Function | File | Responsibility |
|---|---|---|
| `PreflightChecker` | `src/utils/preflight.py` | Validates required packages before import |
| `ResizePad` | `src/preprocessing/transforms.py` | Aspect-ratio-safe resize + symmetric padding |
| `TransformBuilder` | `src/preprocessing/transforms.py` | Builds train/eval transform pipelines |
| `DatasetLoader` | `src/preprocessing/dataset_loader.py` | Loads ImageFolder splits, runs leakage check |
| `LeakageChecker` | `src/preprocessing/leakage.py` | Detects file overlap between splits (pre-existing) |
| `MixupAugmentation` | `src/training/mixup.py` | Batch-level Mixup with blended loss |
| `WarmupCosine` | `src/training/callbacks.py` | Linear warmup + cosine annealing LR schedule |
| `EarlyStopping` | `src/training/callbacks.py` | Patience-based early stopping on any metric |
| `ConvNeXtClassifier` | `src/models/classifier.py` | Builds model, manages param groups, loads checkpoints |
| `unwrap_model` | `src/models/classifier.py` | Unwraps `nn.DataParallel` |
| `GradCAMPlusPlus` | `src/xai/gradcam.py` | Computes GradCAM++ saliency maps via hooks |
| `get_gradcam_target_layer` | `src/xai/gradcam.py` | Returns canonical target layer for ConvNeXt-Base |
| `GradCAMSaver` | `src/xai/gradcam.py` | Generates and saves overlay figures + JSON report |
| `TrainingPlotter` | `src/xai/visualization.py` | Saves all training curves and evaluation plots |
| `ConvNeXtTrainer` | `src/training/train_classifier.py` | Full training loop: fit, evaluate, save |
| `XAIReporter` | `src/xai/xai_reporter.py` | Generates clinical PDF/HTML reports (pre-existing) |
| `LiveLogger` | `src/utils/io_ops.py` | Thread-safe logging with timestamps (pre-existing) |
| `Seeder` | `src/utils/seed_all.py` | Reproducibility seed management (pre-existing) |

---

## 3. Usage Examples

### Minimal training run

```python
from pathlib import Path
import torch

from src.utils.preflight import PreflightChecker
from src.utils.io_ops import LiveLogger
from src.utils.seed_all import Seeder
from src.preprocessing.dataset_loader import DatasetLoader
from src.training.train_classifier import ConvNeXtTrainer

# Preflight
PreflightChecker().check()

# Reproducibility
Seeder(42).seed_all()

# Logging
log = LiveLogger(Path("./convNext/train.log"))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
loader = DatasetLoader("./data/split", image_size=512, log=log)
train_loader, val_loader, test_loader = loader.build_loaders(
    batch_size=8, num_workers=4
)

# Train
trainer = ConvNeXtTrainer(
    num_classes=4,
    class_names=["BCC", "BKL", "MEL", "NV"],
    out_dir=Path("./convNext"),
    log=log,
    device=device,
    epochs=60,
    warmup_epochs=5,
)
history = trainer.fit(train_loader, val_loader)
split_metrics = trainer.evaluate_and_save(val_loader, test_loader)
```

### Loading a saved checkpoint for inference

```python
import torch
from src.models.classifier import ConvNeXtClassifier

device = torch.device("cuda")
model, class_names, ck = ConvNeXtClassifier.load_checkpoint(
    "convNext/best_convnext_checkpoint.pth", device
)
# model is already in eval() mode
```

### Generating GradCAM++ overlays standalone

```python
from pathlib import Path
from src.xai.gradcam import GradCAMSaver
from src.utils.io_ops import LiveLogger

log = LiveLogger(Path("./gradcam.log"))
saver = GradCAMSaver(
    model=model,
    class_names=["BCC", "BKL", "MEL", "NV"],
    out_dir=Path("./gradcam_output"),
    device=device,
    log=log,
)
reports = saver.run(test_loader, max_samples=40)
```

### Running only transforms

```python
from src.preprocessing.transforms import TransformBuilder, ResizePad
from PIL import Image

train_tf, eval_tf = TransformBuilder.build(image_size=512)
img = Image.open("lesion.jpg").convert("RGB")
tensor = eval_tf(img)  # (3, 512, 512) normalised tensor
```

### Checking optional packages

```python
from src.utils.preflight import PreflightChecker

checker = PreflightChecker()
if checker.available("imagededup"):
    print("pHash deduplication available")
```

---

## 4. Data Structure Expected

The pipeline uses `torchvision.datasets.ImageFolder` and expects the following on-disk layout:

```
data/
  split/
    train/
      BCC/
        image001.jpg
        image002.jpg
        ...
      BKL/
        ...
      MEL/
        ...
      NV/
        ...
    validation/          # or val/ — both are accepted
      BCC/
        ...
      ...
    test/
      BCC/
        ...
      ...
```

Rules:
- All three splits (train, val/validation, test) must have exactly the same class folders in the same order.
- The class folder names become the `class_names` list (sorted alphabetically by default).
- Any file overlap between splits is detected by `LeakageChecker` and causes a hard failure before training starts.
- Images should be RGB JPEG or PNG. Grayscale images are not supported without conversion.

---

## 5. Training Arguments

`ConvNeXtTrainer.__init__` accepts the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_classes` | int | required | Number of output classes |
| `class_names` | list[str] | required | Human-readable class labels |
| `out_dir` | Path | required | Root directory for all outputs |
| `log` | LiveLogger | required | Logger instance |
| `device` | torch.device | required | CUDA or CPU device |
| `epochs` | int | 60 | Maximum training epochs |
| `warmup_epochs` | int | 5 | Epochs with frozen backbone |
| `batch_size` | int | 8 | Per-device batch size |
| `lr_head` | float | 2e-4 | Learning rate for classification head |
| `lr_stage7` | float | 2e-5 | Learning rate for ConvNeXt stage-7 |
| `lr_rest` | float | 5e-6 | Learning rate for remaining backbone |
| `weight_decay` | float | 1e-4 | AdamW weight decay |
| `label_smoothing` | float | 0.1 | CrossEntropyLoss label smoothing |
| `dropout` | float | 0.4 | Head dropout probability |
| `mixup_alpha` | float | 0.3 | Beta distribution parameter for Mixup (0 = disabled) |
| `grad_clip` | float | 1.0 | Gradient norm clipping (0 = disabled) |
| `patience` | int | 12 | Early stopping patience (epochs) |
| `gradcam_samples` | int | 40 | Number of GradCAM++ overlays to generate |
| `num_workers` | int | 4 | DataLoader worker processes |

---

## 6. Output Files Generated

After a full `fit()` + `evaluate_and_save()` run, the output directory contains:

```
convNext/
  best_convnext_checkpoint.pth     # Full checkpoint: weights, optimizer, history
  metrics_summary.json             # Val + test scalar metrics (no arrays)
  train.log                        # Full training log with timestamps

  plots/
    01_loss_accuracy.png           # Train/val loss and accuracy curves
    02_macro_f1.png                # Train/val macro-F1 over epochs
    03_generalisation_gap.png      # Train-val gap with overfit zone
    04_lr_schedule.png             # LR schedule (log scale)
    05_balanced_accuracy.png       # Balanced accuracy curves
    val_cm_counts.png              # Validation confusion matrix (counts)
    val_cm_normalised.png          # Validation confusion matrix (row-normalised)
    val_roc.png                    # Validation ROC curves with AUC
    val_pr_curves.png              # Validation precision-recall curves
    val_per_class_metrics.png      # Val per-class P/R/F1 bar chart
    val_confidence_dist.png        # Val confidence distribution per class
    test_cm_counts.png             # Test confusion matrix (counts)
    test_cm_normalised.png         # Test confusion matrix (row-normalised)
    test_roc.png                   # Test ROC curves with AUC
    test_pr_curves.png             # Test precision-recall curves
    test_per_class_metrics.png     # Test per-class P/R/F1 bar chart
    test_confidence_dist.png       # Test confidence distribution per class

  gradcam/
    gradcam_000_MEL_MEL_OK.png     # Side-by-side: original | pred CAM | true CAM
    gradcam_001_NV_MEL_X.png
    ...
    gradcam_report.json            # JSON with stats for every overlay
```

---

## 7. Audit Methods Overview

The `src/audit/` directory contains four independent data quality audit methods coordinated by `AuditRunner`. These run separately from training and validate the dataset before it is used.

| Method | Description |
|---|---|
| Method 1 | Statistical distribution analysis — class balance, pixel statistics, aspect ratios |
| Method 2 | Perceptual hash deduplication — finds near-duplicate images using pHash (requires `imagededup`) |
| Method 3 | Embedding-space clustering — UMAP/t-SNE visualisation of feature embeddings to detect anomalies (requires `umap-learn`) |
| Method 4 | Label-noise detection — identifies potentially mislabelled samples using cross-validation confidence scores |

Audit results are written to `src/audit/reports/` and do not modify the dataset. They produce findings that a human reviewer must act on.

To run all audit methods:

```python
from src.audit import AuditRunner

runner = AuditRunner(data_root="./data/split", out_dir="./audit_reports")
runner.run_all()
```

---

## 8. XAI / GradCAM++ Explanation

### What is GradCAM++?

GradCAM++ (Chattopadhyay et al., WACV 2018) is a gradient-based visual explanation method. It uses the gradients of a target class score flowing back through the final convolutional layer to produce a spatial heatmap showing which image regions most influenced the prediction.

### Target layer

For ConvNeXt-Base, the target layer is `features[7][2].block[0]` — the depthwise 7x7 convolution in the last block of stage 7. This layer has the richest semantic feature maps while still retaining spatial resolution suitable for overlay visualisation.

Access via `get_gradcam_target_layer(model)`.

### How saliency is computed

1. A forward pass produces logits.
2. The score for the target class is back-propagated.
3. Activations `A` and gradients `g` at the target layer are captured via hooks.
4. Second- and third-order gradient terms weight the spatial positions: `alpha = g^2 / (2*g^2 + g^3 * sum(A) + eps)`.
5. The weighted sum over channels is ReLU-clamped and upsampled to input size.
6. The result is min-max normalised to [0, 1].

### Reading the overlays

Each saved figure (`gradcam/gradcam_NNN_PRED_TRUE_OK|X.png`) shows three panels:
- Left: original dermoscopic image with true label
- Centre: GradCAM++ heatmap for the predicted class (jet colormap, 45% opacity)
- Right: GradCAM++ heatmap for the true class (for comparison when the prediction is wrong)

A green title indicates a correct prediction; red indicates an error.

### The `analyse()` output

`GradCAMPlusPlus.analyse(cam, pred_class)` returns a statistics dictionary:

| Key | Meaning |
|---|---|
| `high_activation_pct` | % pixels with activation >= 0.70 |
| `mid_activation_pct` | % pixels with activation in [0.40, 0.70) |
| `mean_activation` | Mean activation across all pixels |
| `peak_activation` | Maximum activation value |
| `primary_region` | Spatial region of peak focus (e.g. `upper-left`, `central`) |
| `xai_summary` | Human-readable one-sentence explanation |

### Clinical integration

`XAIReporter` (in `src/xai/xai_reporter.py`) wraps GradCAM++ results with ICD-10 codes, risk levels, class morphology descriptions, and next-steps recommendations to produce patient-facing PDF/HTML reports.

---

## 9. Monitoring Integration

The monitoring subsystem (in `monitoring/`) tracks model performance in production via `monitoring/performance_tracker.py`.

Metrics emitted during training that are consumed by the tracker:

- Per-epoch validation macro-F1, accuracy, balanced accuracy, loss
- Best checkpoint epoch and score
- GradCAM++ activation statistics per class (from `gradcam_report.json`)
- Final test set scalar metrics (from `metrics_summary.json`)

To integrate monitoring into a training run:

```python
from monitoring.performance_tracker import PerformanceTracker

tracker = PerformanceTracker(run_id="exp001", out_dir=Path("./monitoring"))
for epoch in range(epochs):
    # ... training ...
    tracker.log_epoch(epoch, train_metrics, val_metrics)
tracker.finalise(test_metrics)
```

The tracker is optional — `ConvNeXtTrainer` does not depend on it directly. Wire it externally in your training script after each epoch using the metrics returned from `_eval_epoch`.

---

## 10. Dependencies

### Required (hard fail on missing)

| Package | Version tested | Install |
|---|---|---|
| `torch` | 2.2+ | `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia` |
| `torchvision` | 0.17+ | included with torch |
| `scikit-learn` | 1.4+ | `conda install scikit-learn -c conda-forge` |
| `Pillow` | 10.0+ | `pip install Pillow` |
| `tqdm` | 4.66+ | `conda install tqdm -c conda-forge` |
| `seaborn` | 0.13+ | `conda install seaborn -c conda-forge` |
| `joblib` | 1.3+ | `pip install joblib` |
| `h5py` | 3.10+ | `conda install h5py -c conda-forge` |
| `numpy` | 1.26+ | included with conda base |
| `matplotlib` | 3.8+ | included with seaborn |

### Optional (degraded functionality if missing)

| Package | Purpose | Install |
|---|---|---|
| `imagededup` | Audit Method 2 — perceptual hash deduplication | `pip install imagededup` |
| `umap-learn` | Audit Method 3 — UMAP embedding visualisation | `pip install umap-learn` |

### Full install

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scikit-learn seaborn tqdm h5py -c conda-forge
pip install Pillow joblib imagededup umap-learn
```

Or via the project requirements file:

```bash
pip install -r requirements.txt
```
