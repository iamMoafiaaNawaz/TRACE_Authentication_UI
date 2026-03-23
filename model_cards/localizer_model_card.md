# TRACE — Skin Lesion Localization
## YOLO-Based Object Detection Pipeline (Exp8)

**TRACE** — *Transformative Research in Automated Clinical Evaluation*
Final Year Project · FAST-NUCES

---

## Table of Contents

1. [Overview](#overview)
2. [Trained Weights](#trained-weights)
3. [YOLO Variants Supported](#yolo-variants-supported)
4. [Pseudo Bounding Box Generation](#pseudo-bounding-box-generation)
5. [YOLO Dataset Construction](#yolo-dataset-construction)
6. [Stable Training Configuration](#stable-training-configuration)
7. [NaN Guard — Preventing Checkpoint Corruption](#nan-guard)
8. [DDP NMS Patch](#ddp-nms-patch)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Overlay Visualisation](#overlay-visualisation)
11. [API Endpoint](#api-endpoint)
12. [MedGemma Clinical Reports](#medgemma-clinical-reports)
13. [Module Reference](#module-reference)
14. [Running the Pipeline](#running-the-pipeline)
15. [Hyperparameter Rationale](#hyperparameter-rationale)
16. [Output Structure](#output-structure)

---

## Overview

Since no manual bounding-box annotations exist for the skin lesion dataset, this pipeline generates **pseudo bounding boxes** from image-processing heuristics (Otsu) or from the Exp7 classification model's localisation map, then trains SOTA YOLO detectors on those pseudo-labels.

Three models are trained in parallel and evaluated on val + test splits. The best model (highest mAP@0.5:0.95 on test) is selected automatically.

**Pipeline flow:**

```
Raw images (classification-split dataset)
        ↓
PseudoBoxGenerator  [Otsu or locmap mode]
        ↓
YoloDatasetBuilder  [images/ + labels/ + dataset.yaml]
        ↓
YoloTrainer × N variants  [YOLOv11x, YOLOv10x, YOLOv9e ...]
        ↓
YoloEvaluator  [val + test mAP / P / R / F1 / per-class AP]
        ↓
OverlaySaver   [side-by-side pseudo-GT vs prediction overlays]
        ↓
MedGemmaService (optional)  [structured clinical reports]
```

---

## Trained Weights

| Variant   | File                                    | mAP@0.5 (test) | Epochs |
|-----------|-----------------------------------------|----------------|--------|
| YOLOv11x  | `weights/yolo/yolov11x_best.pt`         | (trained)      | 100    |

The production checkpoint is `weights/yolo/yolov11x_best.pt`.
Training config: `configs/yolo_exp8.yaml` (lr0=3e-4, amp=False, warmup_bias_lr=0.01, batch=16, imgsz=640, 4 GPUs).

---

## YOLO Variants Supported

Registered in `src/models/yolo_registry.py` → `YoloRegistry`:

| Variant | Weights File | Architecture Highlights |
|---|---|---|
| `yolov11x` | `yolo11x.pt` | C3k2 + C2PSA blocks, highest accuracy |
| `yolov11l` | `yolo11l.pt` | C3k2 + C2PSA, faster than x |
| `yolov10x` | `yolov10x.pt` | NMS-free dual-assignment head |
| `yolov10l` | `yolov10l.pt` | NMS-free, efficient large |
| `yolov9e` | `yolov9e.pt` | GELAN + PGI, best on small data |
| `yolov9c` | `yolov9c.pt` | GELAN + PGI compact |
| `yolov8x` | `yolov8x.pt` | Proven stable baseline |

Default experiment runs: `yolov11x, yolov10x, yolov9e`

### Weight Resolution

`YoloRegistry.resolve(variant, weights_dir)` searches:
1. `<weights_dir>/<weights_filename>`
2. `./<weights_filename>` (current directory)

If not found, prints the exact `ultralytics` download command and raises `FileNotFoundError`.

---

## Pseudo Bounding Box Generation

Since no annotation budget exists, bounding boxes are generated algorithmically. Two modes are available via `PseudoBoxGenerator`:

### Mode 1 — Locmap (preferred)

Uses the localisation head output from the Exp7 classification model:

```python
gen = PseudoBoxGenerator(exp7_dir=Path("./exp7"), device=device, logger=log)
```

The Exp7 proxy model (ResNet-50 backbone + 1×1 localisation head) outputs a `(1, 1, H, W)` activation map. `locmap_box()` thresholds this at 0.35 (configurable), fills holes, and converts the binary mask to a normalised `(cx, cy, bw, bh)` box via `_mask_to_box()`.

### Mode 2 — Otsu (fallback)

When no Exp7 checkpoint is found, dermoscopy-aware Otsu thresholding is used:

**Algorithm:**

1. Run Otsu threshold independently on the **green channel** (best lesion contrast in dermoscopy) and **luminance channel**
2. Try both `mask = channel ≥ threshold` (light lesion) and `mask = channel < threshold` (dark lesion) — 4 candidates total
3. Zero out 5% border pixels on each side → removes hardware vignette ring
4. `binary_fill_holes` → fills specular reflection holes inside the lesion
5. Keep only the **largest connected component** (scipy `label`)
6. Score each candidate: `0.6 × centrality + 0.4 × size_score`
   - `centrality = 1 − 2·max(|cx−0.5|, |cy−0.5|)` — penalises off-centre lesions
   - `size_score = 1 − |fraction − 0.25| / 0.25` — peaks at 25% image area (typical lesion size)
7. Select the highest-scoring candidate
8. Apply `_mask_to_box()` with 4% padding

**Why 0.40 fallback box, not 0.75?**
An oversized default box covering 56% of the image produced `cls_loss = 157.5` at epoch 4 in Exp8 run 1, causing NaN explosion. The `(0.5, 0.5, 0.40, 0.40)` fallback covers ~16% of the image area and is much safer for YOLO's classification branch.

### Box Clamping

All boxes pass through `_clamp()` to guarantee YOLO-valid coordinates:
```python
bw = clip(bw, 0.08, 0.95)
bh = clip(bh, 0.08, 0.95)
cx = clip(cx, bw/2 + 0.01, 1 − bw/2 − 0.01)
cy = clip(cy, bh/2 + 0.01, 1 − bh/2 − 0.01)
```

---

## YOLO Dataset Construction

`YoloDatasetBuilder.build(splits, names)` creates:

```
yolo_dataset/
    images/
        train/   000001_stem.jpg  →  symlink to original (or copy if symlink fails)
        val/
        test/
    labels/
        train/   000001_stem.txt  →  "class_idx cx cy bw bh\n"
        val/
        test/
    dataset.yaml
```

**Symlinks vs copies:** Symlinks are used by default to avoid doubling disk usage. Falls back to `shutil.copy2` on systems without symlink support (e.g. some Windows / network filesystems).

**dataset.yaml format:**
```yaml
path: /absolute/path/to/yolo_dataset
train: images/train
val:   images/val
test:  images/test
nc: 4
names:
  - BCC
  - BKL
  - MEL
  - NV
```

The `--skip_build` flag reuses an existing `dataset.yaml` without regenerating labels, saving significant time when rerunning with different YOLO variants.

---

## Stable Training Configuration

`STABLE_TRAIN_DEFAULTS` in `src/training/train_yolo.py` encodes the fixes applied after the NaN explosion in Exp8 run 1.

### Full defaults table

| Parameter | Value | Previous (broken) | Change rationale |
|---|---|---|---|
| `optimizer` | `AdamW` | — | — |
| `lr0` | `3e-4` | `1e-3` | 3× reduction — more conservative |
| `lrf` | `0.01` | — | — |
| `warmup_bias_lr` | **`0.01`** | **`0.1`** | **THE main NaN cause** — 10× reduction |
| `amp` | **`False`** | **`True`** | fp16 amplifies NaN propagation |
| `box` | `5.0` | `7.5` | Lower gradient magnitude |
| `perspective` | **`0.0`** | **`0.0005`** | Extreme crops produce NaN |
| `mixup` | `0.1` | `0.2` | Reduced noise |
| `degrees` | `15.0` | `30.0` | Reduced rotation range |
| `shear` | `2.0` | `5.0` | Reduced geometric distortion |
| `dropout` | `0.05` | `0.1` | Less aggressive dropout |
| `hsv_s` | `0.4` | `0.5` | Reduced saturation jitter |
| `hsv_v` | `0.3` | `0.4` | Reduced value jitter |
| `label_smoothing` | *(omitted)* | — | Deprecated in ultralytics; harmful for nc=4 |

**Root cause of NaN explosion:**
`warmup_bias_lr = 0.1` is 10× higher than the recommended value. Combined with `amp=True` and the large pseudo-label noise (Otsu boxes are imprecise), this produced runaway gradients in the bias terms during warmup. The cls_loss reached 157.5 at epoch 4 before overflowing to NaN.

### Callbacks registered per model

| Event | Callback | Function |
|---|---|---|
| `on_train_start` | `make_start_callback` | Logs start banner with epoch/batch/imgsz |
| `on_train_epoch_end` | `make_epoch_callback` | Logs loss (box/cls/dfl), P/R/F1/mAP, LR, ETA, tracks best |
| `on_train_epoch_end` | `make_nan_guard_callback` | Detects NaN/Inf — stops after 2 consecutive epochs |
| `on_train_end` | `make_end_callback` | Logs final validation metrics |

---

## NaN Guard

`make_nan_guard_callback` in `src/training/yolo_callbacks.py` prevents **checkpoint corruption** when loss diverges:

```
Epoch N:   NaN in loss_items  →  streak = 1  [warn, continue]
Epoch N+1: NaN in loss_items  →  streak = 2  [copy best.pt → best_pre_nan.pt, raise RuntimeError]
```

The `best_pre_nan.pt` copy ensures you always have a usable checkpoint even if `last.pt` was partially written during the crash. `RuntimeError` is caught by the training loop in `exp8_localization.py`, logged, and the next variant continues.

`purge_corrupt_checkpoints()` in `src/utils/io_ops.py` scans for any existing `last.pt` containing `NaN` or `Inf` weights before training starts, deleting them to prevent stale corrupted checkpoints from being loaded by `--resume`.

---

## DDP NMS Patch

Multi-GPU DDP training spawns worker subprocesses that re-import `ultralytics` fresh. If the GPU does not support `torchvision.ops.nms` natively (some configurations), these workers crash.

**Two-layer fix in `src/utils/nms_patch.py`:**

**Layer 1 — disk patch** (`patch_ultralytics_nms_on_disk`):
Rewrites `ultralytics/utils/nms.py` on disk, wrapping the NMS call in a try/except that falls back to CPU:

```python
try:
    i = torchvision.ops.nms(boxes, scores, iou_thres)
except (NotImplementedError, RuntimeError):
    i = torchvision.ops.nms(boxes.cpu().float(), scores.cpu().float(), iou_thres).to(boxes.device)
```

A `.orig` backup is created on first run. The patch is idempotent (checks for `CPU_NMS_FALLBACK` marker).

**Layer 2 — runtime patch** (`apply_runtime_nms_patch`):
Monkey-patches `torchvision.ops.nms` in the current process with the same try/except. Uses `_cpu_fallback_patched` flag to prevent double-patching.

Both patches are applied at experiment startup before any `model.train()` call. Use `--single_gpu` to bypass DDP entirely if both patches fail.

---

## Evaluation Metrics

`YoloEvaluator.evaluate_all()` calls `model.val()` on val and test splits.
`YoloMetricsExtractor.extract()` reads the ultralytics `Results` object:

| Metric | Source | Description |
|---|---|---|
| `precision` | `result.box.mp` | Mean precision across classes |
| `recall` | `result.box.mr` | Mean recall across classes |
| `f1` | `2·P·R / (P+R+ε)` | Harmonic mean |
| `mAP_50` | `result.box.map50` | mAP at IoU threshold 0.5 |
| `mAP_50_95` | `result.box.map` | mAP averaged over IoU 0.5:0.05:0.95 |
| `per_class_AP` | `result.box.ap` | Per-class AP@0.5 dict |

Summary plots generated by `YoloPlotter` in `evaluation/yolo_plots.py`:
- mAP@0.5 and mAP@0.5:0.95 bar charts (val + test side by side, all variants)
- Precision, Recall, F1 bar charts
- Per-class AP bar chart for the best variant

---

## Overlay Visualisation

`OverlaySaver.save()` in `src/xai/overlays.py` generates N side-by-side figures:

- **Left panel:** original image + dashed pseudo GT box (lime)
- **Right panel:** original image + YOLO predicted box (red) + pseudo GT (faded) + IoU score in title

Box matching: best predicted box = highest IoU with the pseudo GT box.
All overlay metadata saved to `overlay_info.json` including normalised `box_norm` dict for direct MedGemma API consumption.

---

## API Endpoint

The trained YOLOv11x checkpoint is served via the TRACE FastAPI service.

### POST /detect

Detects skin lesions in a single dermoscopy image.

**Weights loaded from:** `weights/yolo/yolov11x_best.pt`

**Request:** `multipart/form-data`

| Field   | Type  | Required | Default | Description               |
|---------|-------|----------|---------|---------------------------|
| `image` | file  | Yes      | —       | JPEG / PNG / BMP / TIFF   |
| `conf`  | float | No       | 0.25    | Confidence threshold       |
| `iou`   | float | No       | 0.70    | IoU NMS threshold          |

**Response — 200 OK:**

```json
{
  "request_id": "uuid4",
  "model": "YOLOv11x",
  "image_width": 224,
  "image_height": 224,
  "num_detections": 1,
  "detections": [
    {
      "detection_id": 0,
      "class_id": 2,
      "class_name": "MEL",
      "confidence": 0.909,
      "box": {
        "x1": 29.0, "y1": 29.0, "x2": 185.0, "y2": 177.0,
        "cx_norm": 0.4777, "cy_norm": 0.4598,
        "w_norm": 0.6964, "h_norm": 0.6607
      },
      "class_info": {
        "full_name": "Melanoma",
        "risk": "Critical",
        "icd10": "C43.9",
        "action": "URGENT: Immediate oncology referral"
      }
    }
  ],
  "conf_threshold": 0.25,
  "iou_threshold": 0.7,
  "inference_time_ms": 143.2,
  "warning": "AI output is for research assistance only."
}
```

### POST /detect/batch

Upload up to 32 images. Returns `{"total_images": N, "results": [...]}`.

### Start the API server

```bash
python main_api.py --reload       # development
python main_api.py --workers 4    # production (do not use --reload)
```

Interactive docs: `http://localhost:8000/docs`

---

## MedGemma Clinical Reports

`MedGemmaService` in `src/models/medgemma.py` wraps `google/medgemma-4b-it` for on-demand structured clinical reports.

### Service (offline batch mode)

```python
svc = MedGemmaService(
    model_id="google/medgemma-4b-it",
    cache_dir="/path/to/hf_cache",
    use_4bit=True,          # 4-bit quantised to fit alongside YOLO on single GPU
    local_files_only=True,  # HPC environments with no internet
    max_new_tokens=400,
)
result = svc.generate_report(
    image_path="patient.jpg",
    pred_class="MEL",
    pred_conf=0.87,
    box_cx=0.52, box_cy=0.48,
    box_w=0.31,  box_h=0.28,
)
```

The prompt provides pixel-level box coordinates, relative size, and requests a 6-section structured report: Lesion Location, Morphological Features, Dermoscopic Criteria, Risk Assessment, Recommended Action, AI Limitations.

### REST API (on-demand, doctor-facing)

`MedGemmaAPI` exposes `MedGemmaService` as an HTTP endpoint. MedGemma is lazy-loaded on the first request. Tries Flask first; falls back to Python stdlib `http.server`.

```bash
# Start API server (separate SLURM job or terminal)
python experiments/serve_medgemma.py --port 8787

# Doctor sends a POST request
curl -X POST http://HOST:8787/report \
  -F "image=@patient.jpg" \
  -F "pred_class=MEL" \
  -F "pred_conf=0.87" \
  -F "box_cx=0.52" -F "box_cy=0.48" \
  -F "box_w=0.31"  -F "box_h=0.28"

# Health check
curl http://HOST:8787/health
```

**Response JSON keys:** `pred_class`, `pred_conf`, `box_pixels` (x1/y1/x2/y2), `image_size` (W/H), `report` (full clinical text), `gen_time_sec`.

---

## Module Reference

```
src/
├── models/
│   ├── yolo_registry.py   YoloRegistry
│   │                        .resolve(variant, weights_dir) → Path
│   │                        .list_variants() → List[str]
│   │                        .parse_models_arg("yolov11x,yolov10x") → List[str]
│   ├── pseudo_box.py      PseudoBoxGenerator(exp7_dir, device, logger)
│   │                        .get_box(img_path) → (cx, cy, bw, bh)
│   │                        .mode  [property: "locmap" | "otsu"]
│   │                      otsu_box(img_path) → (cx, cy, bw, bh)
│   │                      locmap_box(lm, thresh) → (cx, cy, bw, bh)
│   └── medgemma.py        MedGemmaService(model_id, cache_dir, ...)
│                            .generate_report(image_path, pred_class, ...) → dict
│                            .unload()
│                          MedGemmaAPI(service, host, port)
│                            .serve()   [blocks]
│
├── training/
│   ├── train_yolo.py      YoloTrainer(variant, weights, yaml, out_dir, ...)
│   │                        .train() → Path  [best checkpoint]
│   │                      STABLE_TRAIN_DEFAULTS  [dict]
│   ├── yolo_callbacks.py  make_start_callback(variant, log)
│   │                      make_epoch_callback(variant, log, total_epochs)
│   │                      make_nan_guard_callback(variant, log)
│   │                      make_end_callback(variant, log)
│   └── yolo_dataset.py    YoloDatasetBuilder(yolo_root, gen, log)
│                            .build(splits, names) → yaml_path
│                          scan_class_folder(root) → (records, names)
│                          load_splits(split_root, data_root, ...) → (tr, va, te, names)
│
├── utils/
│   └── nms_patch.py       patch_ultralytics_nms_on_disk() → bool
│                          apply_runtime_nms_patch()
│
└── xai/
    └── overlays.py        OverlaySaver(best_pt, gen, out_dir, ...)
                             .save(records, class_names, n) → List[dict]
                           xyxy(cx, cy, w, h, W, H) → np.ndarray
                           iou(a, b) → float

evaluation/
├── evaluate_model.py      YoloEvaluator(class_names, log)
│                            .evaluate_split(best_pt, yaml, ...) → dict
│                            .evaluate_all(best_pt, yaml, ...) → {split: dict}
├── metrics.py             YoloMetricsExtractor(class_names)
│                            .extract(ultralytics_result) → dict
│                          compute_f1_from_pr(precision, recall) → float
└── yolo_plots.py          YoloPlotter.save_plots(results, out_dir)
                           YoloPlotter.per_class_ap(results, variant, out_dir)
```

---

## Running the Pipeline

### Standard 3-model run (recommended)

```bash
python exp8_localization.py \
    --data_root    /path/to/dataset_split \
    --output_dir   ./ExperiemntNo8 \
    --weights_dir  /path/to/downloaded/weights \
    --exp7_ckpt_dir ./ExperiemntNo7 \
    --models       yolov11x,yolov10x,yolov9e \
    --epochs       100 \
    --imgsz        640 \
    --batch        16 \
    --patience     30 \
    --n_overlays   50
```

### Single GPU (bypass DDP)

```bash
python exp8_localization.py \
    --single_gpu \
    --models yolov11x \
    --epochs 100
```

### Skip dataset rebuild (reuse existing labels)

```bash
python exp8_localization.py \
    --skip_build \
    --yolo_dataset_dir ./ExperiemntNo8/yolo_dataset \
    --models yolov11x,yolov10x
```

### Otsu-only (no exp7 checkpoint)

If `--exp7_ckpt_dir` is not found, `PseudoBoxGenerator` automatically falls back to Otsu mode — no flags needed.

### Start MedGemma API

```bash
python exp8_localization.py \
    --serve_medgemma \
    --medgemma_port 8787 \
    --medgemma_id   /path/to/medgemma-4b-it
```

---

## Hyperparameter Rationale

### Why AdamW over SGD?

Skin lesion datasets are small relative to YOLO's parameter count. AdamW's adaptive learning rates converge faster and more stably on small batches (16 images). SGD requires careful momentum tuning that is brittle with noisy pseudo-labels.

### Why lr0 = 3e-4?

The original `lr0 = 1e-3` combined with `warmup_bias_lr = 0.1` created a 100× effective learning rate spike in bias terms during warmup. Reducing to `3e-4` gives a 33× spike at worst — still aggressive but stable.

### Why amp = False?

`float16` reduces the representable range of gradients. With noisy Otsu pseudo-labels producing occasional high-loss batches, fp16 overflows to NaN before the loss can recover. Full fp32 is ~15–20% slower but eliminates the risk entirely.

### Why perspective = 0.0?

`perspective = 0.0005` is ultralytics' default. On extreme crops of small lesions, perspective warp can produce degenerate boxes with near-zero area, causing `log(0)` in the DFL loss. Setting to 0.0 removes this source of NaN entirely.

### Why close_mosaic = 10?

Mosaic augmentation combines 4 images randomly. During the last 10 epochs, mosaic is disabled so the model trains on single clean images — improving final mAP on the test distribution.

---

## Output Structure

```
ExperiemntNo8/
├── exp8_log.txt                    Full training log
├── exp8_summary.json               Classes, best model, eval results, config
├── overlay_info.json               Per-sample overlay metadata + box_norm
│
├── yolo_dataset/
│   ├── images/train/ val/ test/    Symlinked image files
│   ├── labels/train/ val/ test/    YOLO label .txt files
│   └── dataset.yaml
│
├── yolo_runs/
│   ├── yolov11x/
│   │   └── weights/
│   │       ├── best.pt             Best checkpoint (monitored: mAP@0.5)
│   │       ├── last.pt             Last epoch checkpoint
│   │       └── best_pre_nan.pt     NaN guard safe copy (if triggered)
│   ├── yolov10x/
│   └── yolov9e/
│
├── eval/
│   ├── yolov11x_val/               ultralytics val output
│   ├── yolov11x_test/
│   └── ...
│
├── overlays/
│   ├── overlay_000.png             Side-by-side GT vs prediction
│   ├── overlay_001.png
│   └── ...
│
├── plots/
│   ├── summary_mAP_50.png          Bar chart all variants val+test
│   ├── summary_mAP_50_95.png
│   ├── summary_precision.png
│   ├── summary_recall.png
│   ├── summary_f1.png
│   └── per_class_ap_best.png       Per-class AP for best model
│
└── medgemma_reports.json           MedGemma clinical reports (if enabled)

weights/
└── yolo/
    └── yolov11x_best.pt    ← production checkpoint (copy of best.pt)
```

---

## Dependencies

```
torch >= 2.1.0
torchvision >= 0.16.0
ultralytics >= 8.0.0        # YOLOv8/9/10/11
scikit-learn >= 1.3.0
Pillow >= 10.0.0
numpy >= 1.24.0
scipy >= 1.11.0             # Otsu: binary_fill_holes, label
matplotlib >= 3.7.0
tqdm >= 4.65.0
fastapi >= 0.110.0
uvicorn[standard] >= 0.27.0
pydantic >= 2.0.0

# Optional
transformers >= 4.40.0      # MedGemma service
accelerate >= 0.27.0        # MedGemma device_map="auto"
bitsandbytes >= 0.41.0      # MedGemma 4-bit quantisation
flask >= 3.0.0              # MedGemma REST API (stdlib fallback available)
```

---

> **Clinical Decision Support:** TRACE localisation is a deployed clinical decision support tool. Detection bounding boxes and MedGemma reports are designed to assist qualified dermatologists in lesion assessment — they augment clinical judgement and do not replace it. Final diagnosis and treatment decisions remain the responsibility of the treating clinician.
