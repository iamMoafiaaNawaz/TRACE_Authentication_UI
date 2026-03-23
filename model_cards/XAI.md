# TRACE — Explainability (XAI)
## GradCAM++, Structured Reports & Clinical Interpretation

**TRACE** — *Transformative Research in Automated Clinical Evaluation*  
Final Year Project · FAST-NUCES

---

## Table of Contents

1. [Overview](#overview)
2. [GradCAM++ — Theory](#gradcam----theory)
3. [Target Layer Selection](#target-layer-selection)
4. [Implementation Details](#implementation-details)
5. [XAI Analyser — Spatial Statistics](#xai-analyser----spatial-statistics)
6. [XAI Reporter — Structured Clinical Language](#xai-reporter----structured-clinical-language)
7. [Rule-Based Clinical Reports](#rule-based-clinical-reports)
8. [MedGemma Clinical Reports](#medgemma-clinical-reports)
9. [Standalone XAI Script](#standalone-xai-script)
10. [YOLO Overlay Visualisation](#yolo-overlay-visualisation)
11. [Module Reference](#module-reference)
12. [Output Files](#output-files)

---

## Overview

TRACE implements a three-layer XAI stack for both the classification (ConvNeXt-Base) and localization (YOLO) pipelines:

```
Layer 1 — Saliency Map:   GradCAMPlusPlus
                              ↓
Layer 2 — Spatial Stats:  XAIAnalyser
                              ↓
Layer 3 — Clinical Text:  XAIReporter → ClinicalReporter → MedGemmaService
```

For classification, XAI answers *"what did the model look at?"*  
For localization, XAI answers *"where is the lesion and how confident is the detection?"*

---

## GradCAM++ — Theory

GradCAM++ (Chattopadhyay et al., WACV 2018) improves upon vanilla GradCAM by using **second and third-order gradient terms** to compute spatially non-uniform channel weights.

### Vanilla GradCAM weights

```
w_k = (1/Z) Σ_{ij} (∂y^c / ∂A^k_{ij})
```

This global average pools the gradients — it under-weights channels where the gradient is strong only in a localised patch (common for small lesions).

### GradCAM++ weights

```
α^k_{ij} = (∂²y^c / ∂(A^k_{ij})²) / [2·(∂²y^c / ∂(A^k_{ij})²) + Σ_{ab} A^k_{ab}·(∂³y^c / ∂(A^k_{ij})³)]

w^k = Σ_{ij} α^k_{ij} · ReLU(∂y^c / ∂A^k_{ij})
```

In practice, using the chain rule and the fact that ReLU is the only nonlinearity after the target layer:

```
g   = ∂y^c / ∂A^k   (gradients at target layer)
g²  = g element-wise squared
g³  = g element-wise cubed

denom   = 2·g² + g³ · Σ(A)   [Σ is spatial sum]
alpha   = g² / (denom + ε)
weights = Σ(alpha · ReLU(g))  [spatial sum]
```

The final CAM:
```
CAM = ReLU(Σ_k w^k · A^k)
```

Bilinearly upsampled to input resolution and normalised to [0, 1].

**Why GradCAM++ over GradCAM for dermoscopy?**  
Melanoma lesions often have a focal malignant region (e.g., atypical vascular pattern in one quadrant) surrounded by benign tissue. GradCAM++ correctly localises this focal region; vanilla GradCAM smears the attribution globally.

---

## Target Layer Selection

For **ConvNeXt-Base**:

```
model._backbone.features[7][2].block[0]
```

This is the **last depthwise 7×7 convolution in stage-7, block-2** — the deepest spatial feature map before the global average pooling layer.

**Why this layer?**

| Property | Value |
|---|---|
| Spatial resolution | 16×16 (for 512×512 input) |
| Channel depth | 1024 |
| Receptive field | Near-full image |
| Semantic level | Highest — abstract morphological features |

Deeper layers (closer to the GAP) encode more semantics; shallower layers encode more texture. For dermoscopy, semantic features (border irregularity, colour pattern, surface texture) are more informative than low-level edges.

**DataParallel safety:**  
The property `ConvNeXtClassifier.gradcam_target_layer` strips any `DataParallel` wrapper before accessing `features[7][2].block[0]`, preventing `AttributeError` on multi-GPU setups.

---

## Implementation Details

`GradCAMPlusPlus` in `src/xai/gradcam.py` uses **PyTorch hooks**:

```python
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        # Forward hook: captures activations A^k
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach())
        )
        # Full backward hook: captures gradients ∂y^c/∂A^k
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach())
        )
```

**`register_full_backward_hook` vs `register_backward_hook`:**  
`register_backward_hook` is deprecated in PyTorch ≥ 2.0 and gives incorrect gradients for modules with multiple inputs. `register_full_backward_hook` receives `grad_input` and `grad_output` as tuples — `go[0]` is the gradient w.r.t. the layer's output tensor, which is what we need.

**Inference-time usage:**

```python
# Two passes per image: once for pred class, once for true class
cam_pred = cam_gen(x.clone(), class_idx=pred_idx)   # gradients for predicted class
cam_true = cam_gen(x.clone(), class_idx=true_idx)   # gradients for true class
```

Each call triggers a forward + backward pass. `retain_graph=False` (default) ensures the computation graph is freed after each backward — important for memory when processing 40+ images.

**Hook cleanup:**  
Always call `cam_gen.remove()` after use. Unhooking prevents memory leaks from dangling references to the activation/gradient tensors.

---

## XAI Analyser — Spatial Statistics

`XAIAnalyser.analyse(cam, pred_class)` in `src/xai/gradcam.py` converts a raw `(H, W)` heatmap to a structured statistics dict:

### Activation thresholds

| Category | Threshold | Meaning |
|---|---|---|
| High activation | ≥ 0.70 | Strongly attended pixels |
| Mid activation | [0.40, 0.70) | Moderately attended pixels |
| Background | < 0.40 | Not attended |

### Region localisation

The centroid of all pixels with activation ≥ 0.70 is computed:

```python
ys, xs = np.where(cam >= 0.70)
cy = ys.mean() / H   # normalised vertical position
cx = xs.mean() / W   # normalised horizontal position
```

Spatial labels:
- Vertical: `upper` (cy < 0.4), `central` (0.4 ≤ cy ≤ 0.6), `lower` (cy > 0.6)
- Horizontal: `left` (cx < 0.4), `central`, `right` (cx > 0.6)
- Combined: e.g. `upper-left`, `central` (when both vertical and horizontal are central)

### Attention strength

| Peak activation | Strength | Interpretation |
|---|---|---|
| > 0.8 | Strong | Model attending confidently to localised feature |
| 0.5 – 0.8 | Moderate | Some focus, likely texture-level features |
| < 0.5 | Weak | Diffuse attention, global statistics |

### Output dict keys

```python
{
    "high_activation_pct":  float,   # % pixels above 0.70
    "mid_activation_pct":   float,   # % pixels in [0.40, 0.70)
    "mean_activation":      float,   # mean heatmap value
    "peak_activation":      float,   # max heatmap value
    "primary_region":       str,     # e.g. "upper-left", "central", "diffuse"
    "attention_strength":   str,     # "Strong" | "Moderate" | "Weak"
    "xai_summary":          str,     # one-line human-readable summary
}
```

---

## XAI Reporter — Structured Clinical Language

`XAIReporter.generate(sample)` in `src/xai/xai_reporter.py` enriches the raw spatial stats with clinically-oriented interpretation:

### Attention quality assessment

Three-tier system with explanatory notes:

**Strong** (`peak ≥ 0.85` and `high_pct ≥ 15%`):
> "The model demonstrates highly focused, confident attention on a localised region of the lesion. This is consistent with detection of a specific morphological feature (e.g., rolled border in BCC, colour variegation in MEL)."

**Moderate** (`peak ≥ 0.60` and `high_pct ≥ 5%`):
> "The model shows moderate spatial focus. Attention is present but somewhat distributed. This may reflect reliance on texture-level features rather than a single morphological landmark."

**Weak / Diffuse** (all other cases):
> "Activation is diffuse across the image with no clear focal point. The model may be relying on global texture or colour statistics. Consider reviewing this prediction manually."

### Decision margin analysis

```python
decision_margin = top1_confidence - top2_confidence
```

| Margin | Assessment |
|---|---|
| ≥ 0.50 | Very high — unambiguous classification |
| ≥ 0.25 | High — confident but some uncertainty |
| ≥ 0.10 | Moderate — consider top2 as differential |
| < 0.10 | Low — borderline case, manual review recommended |

### Region clinical notes

Each spatial region maps to a dermoscopy-relevant clinical note:

| Region | Clinical note |
|---|---|
| `central` | Central lesion body — core morphological features densest here |
| `upper-central` | May correspond to lesion apex or elevated nodule |
| `upper-left` / `upper-right` | Could indicate asymmetric spread |
| `lower-left` / `lower-right` | May reflect border irregularity |
| `diffuse` | No dominant focus — model using global image statistics |

### Expected morphological features

For each predicted class, the reporter lists the dermoscopic features the model *should* be attending to, allowing clinicians to verify whether the attention map is anatomically plausible:

| Class | Key Features |
|---|---|
| BCC | Pearly nodule, rolled border, telangiectasia, waxy surface |
| MEL | Irregular scalloped border, colour variegation, regression zones, satellite lesions |
| BKL | Stuck-on waxy appearance, horn cysts, rough verrucous surface |
| NV | Symmetric round/oval, regular borders, uniform colour, stable size |

### MedGemma prompt builder

`XAIReporter.build_medgemma_prompt(sample, xai_report)` constructs a structured user-turn prompt including:
- Model prediction, confidence, ICD-10 code, ground truth
- Full class probability distribution
- GradCAM++ statistics (attention quality, region, peak, high activation %, margin)
- Known morphological features for the predicted class

The system prompt (`MEDGEMMA_SYSTEM_PROMPT`) instructs the model to generate a 6-section structured report under 400 words.

---

## Rule-Based Clinical Reports

`ClinicalReporter.generate(sample)` in `evaluation/reporter.py` produces a deterministic structured report without any LLM:

| Key | Content |
|---|---|
| `model_prediction` | Predicted class (e.g. "MEL") |
| `prediction_confidence` | Confidence + level (e.g. "87.3% (very high)") |
| `icd10` | ICD-10 code |
| `risk_level` | Low / High / Critical |
| `findings` | One-line clinical summary |
| `differential_diagnosis` | Top-3 classes with probabilities + descriptions |
| `xai_interpretation` | Region + activation summary |
| `recommended_next_step` | Full clinical action text |
| `disclaimer` | Clinical decision support notice |

**Risk levels and actions per class:**

| Class | Risk | Recommended Action |
|---|---|---|
| MEL | Critical | URGENT same-day dermatology referral; excisional biopsy |
| BCC | High | Refer within 2 weeks; excision with 4mm margins |
| BKL | Low | Reassure patient; monitor for rapid change |
| NV | Low | Routine surveillance; ABCDE rule at follow-up |

---

## MedGemma Clinical Reports

`generate_medgemma_reports()` in `trace_convnext.py` (and `MedGemmaService` in `src/models/medgemma.py`) generates free-text clinical reports using `google/medgemma-4b-it`.

### Requirements

```bash
pip install transformers accelerate bitsandbytes
export HF_TOKEN=hf_your_token_here   # request access at HuggingFace
```

### Loading strategy

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

4-bit NF4 quantisation reduces VRAM from ~16GB to ~5GB, allowing MedGemma to coexist with ConvNeXt on a single A100.

### Prompt structure

```
SYSTEM: You are a medical AI assistant specialising in dermatology...
        [6-section structured report instructions]

USER:   SKIN LESION AI CLASSIFICATION — TRACE ConvNeXt-Base

        === MODEL PREDICTION ===
        Predicted class:     MEL
        Confidence:          87.3%
        Ground truth:        MEL  (CORRECT)
        ICD-10:              C43 (Malignant melanoma of skin)

        === CLASS PROBABILITIES ===
        MEL: 87.3%  |  BCC: 6.1%  |  NV: 4.8%  |  BKL: 1.8%

        === XAI ANALYSIS (GradCAM++) ===
        Attention quality:   Strong
        Primary focus:       central
        Peak activation:     0.9124
        High activation pct: 31.4%
        Decision margin:     81.2%

        === KNOWN MORPHOLOGICAL FEATURES FOR MEL ===
        Asymmetric shape with irregular scalloped border...
```

### Greedy decoding

`do_sample=False` ensures **deterministic output** for reproducibility and FYP defence demonstration. The same image always produces the same report.

### Graceful fallback

If `transformers` is not installed or `HF_TOKEN` is missing, the function returns a list of dicts with `"medgemma_report": "skipped_no_transformers"` or `"skipped_no_hf_token"` — the rest of the pipeline continues unaffected.

---

## Standalone XAI Script

`evaluation/xai_gradcam.py` runs GradCAM++ on a saved checkpoint without any training infrastructure. Designed specifically for the scenario where the training run crashed **after** saving `best_convnext_checkpoint.pth`.

### Dataset mode

```bash
python evaluation/xai_gradcam.py \
    --checkpoint ./convNext/best_convnext_checkpoint.pth \
    --data_root  /path/to/dataset_split \
    --output_dir ./xai_results \
    --split      test \
    --num_samples 40 \
    --image_size 512
```

Generates per sample:
- 4-panel figure: Original | GradCAM++ pred | GradCAM++ true | Class probability bar chart
- `gradcam_report.json` with all per-sample stats
- `summary_stats.png` — per-class accuracy + confidence distribution histogram

### Single image mode

```bash
python evaluation/xai_gradcam.py \
    --checkpoint ./convNext/best_convnext_checkpoint.pth \
    --single_image /path/to/patient.jpg \
    --output_dir ./xai_results
```

Generates a 3-panel figure (no ground truth panel) and `single_image_report.json` including ICD-10 code, risk level, and recommended action.

### Checkpoint loading (`CheckpointLoader`)

Handles three edge cases that arise from interrupted training runs:

1. **DataParallel prefix** — strips `module.` from all state dict keys
2. **Missing keys** — `strict=False` load, missing keys logged but not fatal
3. **Old checkpoints** — `class_names` and `dropout` defaulted if not present

### Four-panel figure layout

```
[Original Image]  [GradCAM++ → Pred Class]  [GradCAM++ → True Class]  [Class Probabilities]

Title: CORRECT/WRONG | Pred: MEL (87.3%) | True: MEL
       (green for correct, red for wrong)
```

The probability bar chart uses colour coding:
- Blue — predicted class
- Green — true class (only visible when different from prediction)
- Grey — other classes

A vertical dashed line at 0.5 marks the decision boundary.

---

## YOLO Overlay Visualisation

`OverlaySaver` in `src/xai/overlays.py` produces a different type of visualisation focused on **localisation quality**:

```
[Left panel]                    [Right panel]
Original + pseudo GT box        Original + predicted box (red)
(lime dashed)                   + pseudo GT box (faded lime)
                                + IoU score in title
```

The IoU between pseudo GT and prediction is computed via:
```python
def iou(a, b):  # a, b are xyxy pixel arrays
    inter = max(0, min(a[2],b[2]) - max(a[0],b[0])) × max(0, min(a[3],b[3]) - max(a[1],b[1]))
    union = area_a + area_b - inter
    return inter / (union + ε)
```

The best predicted box per sample is the one with the **highest IoU to the pseudo GT** (not the highest confidence). This is more meaningful when evaluating whether the model found the correct lesion region.

---

## Module Reference

```
src/xai/
├── gradcam.py        GradCAMPlusPlus(model, target_layer)
│                       .__call__(x, class_idx=None) → np.ndarray (H, W) in [0,1]
│                       .remove()
│                     XAIAnalyser
│                       .analyse(cam, pred_class) → dict
│                       HIGH_THRESH = 0.70
│                       MID_THRESH  = 0.40
├── overlays.py       OverlaySaver(best_pt, gen, out_dir, device_list, imgsz, log)
│                       .save(records, class_names, n=50) → List[dict]
│                     xyxy(cx, cy, w, h, W, H) → np.ndarray
│                     iou(a, b) → float
└── xai_reporter.py   XAIReporter
                        .generate(sample) → dict
                        .build_medgemma_prompt(sample, xai_report) → str
                      MEDGEMMA_SYSTEM_PROMPT  [str constant]
                      CLASS_DESCRIPTIONS, CLASS_MORPHOLOGY
                      EXPECTED_FEATURES, RISK_LEVELS
                      ICD10_CODES, NEXT_STEPS

evaluation/
├── xai_gradcam.py    CheckpointLoader(path, device).load() → (model, class_names)
│                     GradCAMPlusPlus  [self-contained copy]
│                     ClinicalReportGenerator.generate(pred, true, conf, ...) → dict
│                     run_on_dataset(model, class_names, device, ...) → List[dict]
│                     run_on_single_image(model, class_names, ...) → dict
│                     save_summary_plots(reports, class_names, output_dir)
└── reporter.py       ClinicalReporter.generate(sample) → dict
```

---

## Output Files

### Classification XAI (`convNext/gradcam/`)

| File | Contents |
|---|---|
| `gradcam_NNNN_<pred>_<true>_OK/X.png` | 4-panel overlay image per sample |
| `gradcam_report.json` | Raw GradCAM++ data for all samples |
| `xai_reports.json` | Structured XAI reports (attention quality, margin, features) |
| `xai_reports.txt` | Human-readable XAI text for FYP presentation |
| `clinical_reports.json` | Rule-based clinical reports (structured) |
| `clinical_reports.txt` | Human-readable clinical text |
| `medgemma_reports.json` | MedGemma free-text reports (if enabled) |
| `medgemma_reports.txt` | Human-readable MedGemma text |

### Standalone XAI (`xai_results/`)

| File | Contents |
|---|---|
| `NNNN_<pred>_<true>_OK/X.png` | 4-panel overlay |
| `gradcam_report.json` | All sample data |
| `summary_stats.png` | Per-class accuracy + confidence histogram |
| `single_<pred>_<conf>.png` | Single-image 3-panel figure |
| `single_image_report.json` | Single-image report with ICD-10, risk, action |

### YOLO XAI (`ExperiemntNo8/overlays/`)

| File | Contents |
|---|---|
| `overlay_NNN.png` | 2-panel GT vs prediction overlay |
| `overlay_info.json` | Per-sample: pred/true class, conf, IoU, box_norm |

---

## FYP Defence Talking Points

**On GradCAM++:**  
> "We used GradCAM++ rather than vanilla GradCAM because lesion features in dermoscopy are often focal — a single asymmetric region or colour spot. GradCAM++'s second and third-order gradient terms correctly localise focal activations without spreading attribution globally."

**On target layer choice:**  
> "We hook into `features[7][2].block[0]` — the last depthwise convolution in ConvNeXt-Base's final stage — because it provides the highest semantic resolution (16×16 spatial map, 1024 channels) before global average pooling collapses all spatial information."

**On the three-layer XAI stack:**  
> "Our XAI pipeline has three layers: raw saliency maps, spatial statistics (attention quality, region, peak activation), and clinical language (morphological feature matching, decision margin, ICD-10 codes). This bridges the gap between gradient heatmaps and the information a dermatologist actually needs."

**On MedGemma:**  
> "For the highest-confidence predictions, we optionally run MedGemma-4B-IT — a multimodal medical LLM — to generate structured clinical reports combining the model's prediction with the GradCAM++ region analysis. This provides a clinician with a complete AI-assisted assessment rather than just a label and a heatmap."

---

> **Clinical Decision Support:** TRACE XAI outputs — saliency maps, spatial statistics, clinical reports, and MedGemma text — are components of a deployed clinical decision support system. They are designed to give dermatologists structured, interpretable evidence to support their assessment. Final diagnosis and treatment decisions remain the responsibility of the treating clinician.