# Model Card — TRACE YOLO Skin Lesion Localizer (Exp8)

## Model Details

| Property | Value |
|---|---|
| **Model name** | TRACE YOLO Skin Lesion Localizer |
| **Variants** | YOLOv11x, YOLOv10x, YOLOv9e (default ensemble candidates) |
| **Task** | Object detection — lesion bounding box + classification |
| **Input resolution** | 640 × 640 |
| **Output classes** | 4 (BCC, BKL, MEL, NV) |
| **Framework** | ultralytics ≥ 8.0.0 / PyTorch ≥ 2.1.0 |
| **Developed by** | FAST-NUCES Final Year Project Team |

## Intended Use

**Primary use:** Lesion localisation within dermoscopy images to complement the ConvNeXt-Base classifier. Provides bounding box coordinates for the detected lesion region, which are used to generate structured MedGemma clinical reports with pixel-level location context.

**Intended users:** Qualified dermatologists using the TRACE system; system integrators connecting the bounding box output to downstream report generation.

**Out-of-scope uses:** Detecting multiple lesions per image (single primary lesion assumed); use on non-dermoscopy images.

## Training Data

- **Source:** Same ISIC dataset as ConvNeXt — classification split repurposed for detection
- **Annotation strategy:** Pseudo bounding boxes — no manual annotations used
  - **Locmap mode** (preferred): bounding box derived from Exp7 classification model's activation map
  - **Otsu mode** (fallback): dermoscopy-aware Otsu thresholding with vignette removal and centrality scoring
- **Label format:** YOLO normalised `(class_idx cx cy bw bh)` per image

## Training Procedure

- **Optimiser:** AdamW, lr0=3e-4 (reduced from default 1e-3 to prevent NaN)
- **Key stability fix:** `warmup_bias_lr=0.01` (default 0.1 caused cls_loss=157.5 NaN at epoch 4)
- **AMP:** Disabled (`amp=False`) — fp16 amplified NaN propagation with noisy pseudo-labels
- **Epochs:** 100, patience=30 (early stopping on mAP@0.5)
- **Augmentation:** Mosaic=0.8, Mixup=0.1, Degrees=15, HSV jitter (reduced from defaults)
- **NaN guard:** 2-consecutive-epoch NaN detection; saves `best_pre_nan.pt` before abort
- **Reproducibility:** Seed=42, deterministic=True

## Evaluation Results

*(Fill in after final training run)*

| Variant | Split | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:95 |
|---|---|---|---|---|---|---|
| yolov11x | val | — | — | — | — | — |
| yolov11x | test | — | — | — | — | — |
| yolov10x | val | — | — | — | — | — |
| yolov10x | test | — | — | — | — | — |
| yolov9e | val | — | — | — | — | — |
| yolov9e | test | — | — | — | — | — |

**Best model** (highest test mAP@0.5:0.95): _to be filled_

## Limitations

- Bounding boxes are derived from pseudo-labels — localisation accuracy is constrained by the quality of Otsu/locmap box generation
- Single detection per image — multi-lesion scenarios not supported
- IoU between pseudo GT and model prediction is the primary quality indicator for localisation reliability

## Ethical Considerations

- Bounding box coordinates are passed to MedGemma to contextualise clinical reports — erroneous boxes may influence report quality
- All YOLO outputs feed into a clinical decision support flow; a dermatologist reviews all reports before clinical action

## How to Load

```python
from ultralytics import YOLO

model   = YOLO("./ExperiemntNo8/yolo_runs/yolov11x/weights/best.pt")
results = model.predict("lesion.jpg", conf=0.25, iou=0.45)
boxes   = results[0].boxes.xyxy   # pixel coordinates
```