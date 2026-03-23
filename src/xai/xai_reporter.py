# -*- coding: utf-8 -*-
"""
src/xai/xai_reporter.py
========================
Enhanced XAI report generation from GradCAM++ output.

Classes
-------
XAIReporter
    Converts a GradCAM++ sample dict into a rich, structured XAI report
    with attention quality, region clinical relevance, decision margin
    analysis, morphological feature expectations, and MedGemma prompt.

Domain knowledge tables used:
    CLASS_MORPHOLOGY     — expected dermoscopic features per class
    ICD10_CODES          — ICD-10 code per class
    RISK_LEVELS          — risk level per class
    NEXT_STEPS           — recommended clinical action per class
"""

from typing import Dict, List

# ==============================================================================
# DOMAIN KNOWLEDGE
# ==============================================================================

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "BCC": "Basal Cell Carcinoma — most common skin cancer, locally invasive, rarely metastatic.",
    "BKL": "Benign Keratosis-like Lesion — seborrhoeic keratosis or solar lentigo, benign.",
    "MEL": "Melanoma — most dangerous skin cancer, high metastatic potential, urgent referral.",
    "NV":  "Melanocytic Nevi — common benign mole, melanocyte proliferation.",
}

CLASS_MORPHOLOGY: Dict[str, str] = {
    "BCC": (
        "Typically presents as a pearly or translucent nodule with rolled edges and "
        "telangiectasia. May show ulceration in advanced cases. Slow-growing, locally destructive."
    ),
    "BKL": (
        "Characterised by a waxy, stuck-on appearance with a rough surface. "
        "Colour varies from tan to dark brown. Well-demarcated borders. "
        "Multiple lesions common in older adults."
    ),
    "MEL": (
        "Asymmetric lesion with irregular, notched or scalloped border. "
        "Colour variegation (brown, black, red, white, blue). Diameter >6 mm common. "
        "May be flat or raised. ABCDE criteria: Asymmetry, Border, Colour, Diameter, Evolution."
    ),
    "NV": (
        "Well-circumscribed, symmetric lesion with regular borders and uniform colour. "
        "Usually <6 mm. Stable over time. Compound, junctional, or intradermal subtypes."
    ),
}

EXPECTED_FEATURES: Dict[str, List[str]] = {
    "BCC": [
        "Pearly/translucent nodular surface",
        "Rolled or raised border with telangiectasia",
        "Central ulceration or erosion (advanced)",
        "Waxy or skin-coloured appearance",
    ],
    "MEL": [
        "Asymmetric shape with irregular scalloped border",
        "Multiple colours (brown, black, red, white, blue)",
        "Diameter typically >6 mm",
        "Regression zones (grey-white areas)",
        "Satellite lesions or satellite pigmentation",
    ],
    "BKL": [
        "Stuck-on waxy appearance",
        "Horn cysts visible under dermoscopy",
        "Well-demarcated border",
        "Rough, verrucous surface texture",
        "Tan to dark brown homogeneous colour",
    ],
    "NV": [
        "Symmetric round or oval shape",
        "Regular, well-defined borders",
        "Uniform tan/brown colour",
        "Stable size and appearance over time",
        "Smooth or slightly raised surface",
    ],
}

RISK_LEVELS: Dict[str, str] = {
    "BCC": "High", "MEL": "Critical", "BKL": "Low", "NV": "Low",
}

ICD10_CODES: Dict[str, str] = {
    "BCC": "C44 (Basal cell carcinoma of skin)",
    "MEL": "C43 (Malignant melanoma of skin)",
    "BKL": "L82 (Seborrhoeic keratosis)",
    "NV":  "D22 (Melanocytic naevi)",
}

NEXT_STEPS: Dict[str, str] = {
    "BCC": (
        "Refer to dermatology within 2 weeks for biopsy and excision planning. "
        "Surgical excision with 4 mm margins is first-line. "
        "Mohs micrographic surgery for high-risk locations (face, ears)."
    ),
    "MEL": (
        "URGENT same-day dermatology referral. Immediate excisional biopsy required "
        "to confirm diagnosis and determine Breslow thickness. Do not delay."
    ),
    "BKL": (
        "Reassure patient — benign lesion. Monitor for rapid morphological change. "
        "Cryotherapy or curettage available if cosmetically desired. Annual skin check."
    ),
    "NV": (
        "Routine skin surveillance. Apply ABCDE rule at each follow-up. "
        "Photograph for baseline comparison. Excise if significant change over time."
    ),
}

_REGION_NOTES: Dict[str, str] = {
    "central":       "Central lesion body — core morphological features densest here.",
    "upper-central": "Upper-central area — may correspond to lesion apex or elevated nodule.",
    "lower-central": "Lower-central area — may include lesion base or perilesional skin.",
    "upper-left":    "Peripheral upper-left quadrant — could indicate asymmetric spread.",
    "upper-right":   "Peripheral upper-right quadrant — could indicate asymmetric spread.",
    "lower-left":    "Peripheral lower-left quadrant — may reflect border irregularity.",
    "lower-right":   "Peripheral lower-right quadrant — may reflect border irregularity.",
    "central-left":  "Left-central region — lateral lesion body.",
    "central-right": "Right-central region — lateral lesion body.",
    "diffuse":       "No dominant focus — model using global image statistics.",
}

# MedGemma system prompt
MEDGEMMA_SYSTEM_PROMPT = (
    "You are a medical AI assistant specialising in dermatology.\n"
    "You are reviewing an automated skin lesion classification result from TRACE.\n"
    "Generate a concise structured clinical interpretation report with:\n"
    "  - FINDINGS, DIFFERENTIAL, XAI INTERPRETATION, RECOMMENDATION, LIMITATIONS.\n"
    "End with: 'This report is AI-generated and must be reviewed by a qualified dermatologist.'\n"
    "Keep total response under 400 words."
)


# ==============================================================================
# XAI REPORTER
# ==============================================================================

class XAIReporter:
    """
    Converts a GradCAM++ sample dict into a structured XAI report.

    Also builds the MedGemma prompt string for LLM-based clinical reports.

    Example
    -------
    >>> reporter = XAIReporter()
    >>> report = reporter.generate(sample)
    >>> prompt = reporter.build_medgemma_prompt(sample, report)
    """

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def generate(self, sample: Dict) -> Dict:
        """
        Generate a rich structured XAI report from a GradCAM++ sample dict.

        Parameters
        ----------
        sample : dict — output from ``Plotter.save_gradcam_samples``

        Returns
        -------
        dict with keys:
            xai_method, target_layer, prediction, confidence,
            attention_quality, attention_note,
            primary_focus_region, region_clinical_note,
            peak_activation, high_activation_pct, mid_activation_pct,
            mean_activation, decision_margin, margin_assessment,
            top2_differential, expected_morphological_features,
            full_probability_distribution, icd10_code, disclaimer
        """
        pred     = sample["pred_class"]
        conf     = sample["pred_confidence"]
        region   = sample.get("primary_region", "unknown")
        peak     = sample.get("peak_activation", 0.0)
        high_pct = sample.get("high_activation_pct", 0.0)
        mid_pct  = sample.get("mid_activation_pct", 0.0)
        mean_act = sample.get("mean_activation", 0.0)
        probs    = sample.get("class_probs", {})

        attention_quality, attention_note = self._assess_attention(peak, high_pct)
        sorted_probs   = sorted(probs.items(), key=lambda x: -x[1])
        decision_margin, margin_note, top2 = self._assess_margin(conf, sorted_probs)

        return {
            "xai_method":           "GradCAM++ (Chattopadhyay et al., 2018)",
            "target_layer":         "ConvNeXt-Base features[7][2].block[0] (last depthwise conv, stage-7)",
            "prediction":           pred,
            "confidence":           f"{conf:.1%}",
            "attention_quality":    attention_quality,
            "attention_note":       attention_note,
            "primary_focus_region": region,
            "region_clinical_note": _REGION_NOTES.get(region, f"Region: {region}"),
            "peak_activation":      round(peak, 4),
            "high_activation_pct":  f"{high_pct:.1f}%",
            "mid_activation_pct":   f"{mid_pct:.1f}%",
            "mean_activation":      round(mean_act, 4),
            "decision_margin":      f"{decision_margin:.1%}",
            "margin_assessment":    margin_note,
            "top2_differential":    top2,
            "expected_morphological_features": EXPECTED_FEATURES.get(pred, []),
            "full_probability_distribution": {
                k: f"{v:.1%}" for k, v in sorted_probs
            },
            "icd10_code":           ICD10_CODES.get(pred, "Unknown"),
            "disclaimer": (
                "TRACE clinical decision support output. "
                "GradCAM++ saliency maps indicate regions of high model attention "
                "and are intended to assist qualified dermatologists. "
                "Final diagnosis and treatment decisions remain the clinician's responsibility."
            ),
        }

    def build_medgemma_prompt(self, sample: Dict, xai_report: Dict) -> str:
        """Build the user-turn text prompt for MedGemma from sample + XAI data."""
        pred    = sample["pred_class"]
        conf    = sample["pred_confidence"]
        true_cls = sample["true_class"]
        correct  = sample.get("correct", False)
        probs    = sample.get("class_probs", {})

        sorted_probs_str = "  |  ".join(
            f"{cls}: {p:.1%}"
            for cls, p in sorted(probs.items(), key=lambda x: -x[1])
        )

        return (
            f"SKIN LESION AI CLASSIFICATION — TRACE ConvNeXt-Base\n\n"
            f"=== MODEL PREDICTION ===\n"
            f"Predicted class:     {pred}\n"
            f"Confidence:          {conf:.1%}\n"
            f"Ground truth:        {true_cls}  ({'CORRECT' if correct else 'INCORRECT'})\n"
            f"ICD-10:              {ICD10_CODES.get(pred, 'Unknown')}\n\n"
            f"=== CLASS PROBABILITIES ===\n"
            f"{sorted_probs_str}\n\n"
            f"=== XAI ANALYSIS (GradCAM++) ===\n"
            f"Attention quality:   {xai_report['attention_quality']}\n"
            f"Primary focus:       {xai_report['primary_focus_region']}\n"
            f"Peak activation:     {xai_report['peak_activation']}\n"
            f"High activation pct: {xai_report['high_activation_pct']}\n"
            f"Decision margin:     {xai_report['decision_margin']}\n"
            f"Region note:         {xai_report['region_clinical_note']}\n\n"
            f"=== KNOWN MORPHOLOGICAL FEATURES FOR {pred} ===\n"
            f"{CLASS_MORPHOLOGY.get(pred, 'Not available')}\n\n"
            f"Please generate a structured clinical interpretation report as instructed."
        )

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_attention(peak: float, high_pct: float):
        if peak >= 0.85 and high_pct >= 15:
            quality = "Strong"
            note = (
                "The model demonstrates highly focused, confident attention on a "
                "localised region of the lesion. This is consistent with detection of "
                "a specific morphological feature (e.g., rolled border in BCC, colour "
                "variegation in MEL)."
            )
        elif peak >= 0.60 and high_pct >= 5:
            quality = "Moderate"
            note = (
                "The model shows moderate spatial focus. Attention is present but "
                "somewhat distributed. May reflect reliance on texture-level features "
                "rather than a single morphological landmark."
            )
        else:
            quality = "Weak / Diffuse"
            note = (
                "Activation is diffuse with no clear focal point. The model may rely "
                "on global texture or colour statistics. Consider reviewing this "
                "prediction manually."
            )
        return quality, note

    @staticmethod
    def _assess_margin(conf: float, sorted_probs):
        top2_class = sorted_probs[1][0] if len(sorted_probs) > 1 else "N/A"
        top2_p     = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        margin     = conf - top2_p

        if margin >= 0.50:
            note = "Very high decision margin — unambiguous classification."
        elif margin >= 0.25:
            note = "High decision margin — confident but some uncertainty exists."
        elif margin >= 0.10:
            note = f"Moderate margin — consider {top2_class} as differential."
        else:
            note = f"Low margin ({margin:.1%}) — borderline case, manual review recommended."

        return margin, note, f"{top2_class} ({top2_p:.1%})"

    def __repr__(self) -> str:
        return "XAIReporter()"