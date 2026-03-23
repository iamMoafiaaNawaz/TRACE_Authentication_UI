# Model Card — TRACE ConvNeXt-Base Skin Lesion Classifier

## Model Details

| Property | Value |
|---|---|
| **Model name** | TRACE ConvNeXt-Base Skin Lesion Classifier |
| **Architecture** | ConvNeXt-Base + 2-layer MLP head |
| **Backbone weights** | ImageNet-1K (`ConvNeXt_Base_Weights.IMAGENET1K_V1`) |
| **Input resolution** | 512 × 512 (ResizePad — aspect-ratio preserving) |
| **Output classes** | 4 (BCC, BKL, MEL, NV) |
| **Parameters** | ~89M total; ~1.5M head-only during warmup |
| **Framework** | PyTorch ≥ 2.1.0 |
| **Developed by** | FAST-NUCES Final Year Project Team |

## Intended Use

**Primary use:** Clinical decision support for dermatologists performing skin lesion assessment. The model analyses dermoscopy images and provides a ranked differential with confidence scores and GradCAM++ attention maps highlighting the regions that drove the prediction.

**Intended users:** Qualified dermatologists and trained clinical staff using the TRACE system as a second-opinion tool.

**Out-of-scope uses:** Autonomous diagnosis without clinician oversight; use on non-dermoscopy images (standard photography, histology); use by patients without clinical interpretation.

## Training Data

- **Dataset:** ISIC (International Skin Imaging Collaboration) archive — dermoscopy images
- **Split:** Pre-divided train / validation / test
- **Classes and labels:**

| Class | ICD-10 | Description |
|---|---|---|
| BCC | C44 | Basal Cell Carcinoma |
| BKL | L82 | Benign Keratosis-like Lesion |
| MEL | C43 | Melanoma |
| NV | D22 | Melanocytic Nevi |

- **Preprocessing:** ResizePad to 512×512 (black zero-padding), ImageNet normalisation
- **Data integrity:** Verified clean via 4-method audit (MD5, pHash, embedding DBSCAN, hard crop probe)

## Training Procedure

- **Phase 1 (warmup):** 5 epochs, backbone frozen, head-only training at lr=2e-4
- **Phase 2 (fine-tune):** Differential LR — head 2e-4, stage-7 2e-5, rest 5e-6
- **Loss:** Weighted cross-entropy (inverse-frequency class weights) + label smoothing ε=0.1
- **Optimiser:** AdamW, weight_decay=1e-4
- **Scheduler:** WarmupCosine (linear ramp → cosine annealing to η_min=1e-7)
- **Regularisation:** Mixup α=0.3, dropout=0.4, RandomErasing p=0.15, gradient clip=1.0
- **Early stopping:** Patience=12 on val macro-F1
- **Reproducibility:** Seed=42, deterministic CUDA ops

## Evaluation Results

*(Fill in after final training run)*

| Split | Accuracy | Balanced Acc | Macro F1 | MCC | AUC-OvR |
|---|---|---|---|---|---|
| Train (eval) | — | — | — | — | — |
| Validation | — | — | — | — | — |
| Test | — | — | — | — | — |

## Limitations

- Performance is dependent on image quality and dermoscopy hardware calibration
- Model was trained on ISIC data; generalisation to other imaging equipment requires validation
- Class balance in training data affects per-class performance — BKL and NV are typically over-represented
- GradCAM++ attention maps are saliency proxies, not pixel-level segmentation masks

## Ethical Considerations

- All predictions must be reviewed by a qualified dermatologist before clinical action
- MEL (melanoma) predictions trigger urgent referral recommendations — these should never be dismissed without clinician review
- The system does not store patient data; inference is stateless

## How to Load

```python
from src.inference import InferencePipeline

pipe   = InferencePipeline("./convNext/best_convnext_checkpoint.pth")
result = pipe.predict_image("/path/to/lesion.jpg")
print(result["prediction"], result["confidence"])
```