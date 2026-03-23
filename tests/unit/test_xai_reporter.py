# -*- coding: utf-8 -*-
"""
tests/unit/test_xai_reporter.py
================================
Unit tests for XAIReporter.

Pure logic — no model, no GPU, no images required.
"""

import pytest

from src.xai.xai_reporter import (
    XAIReporter,
    MEDGEMMA_SYSTEM_PROMPT,
    CLASS_DESCRIPTIONS,
    CLASS_MORPHOLOGY,
    EXPECTED_FEATURES,
    RISK_LEVELS,
    ICD10_CODES,
    NEXT_STEPS,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

def _sample(pred="MEL", conf=0.87, region="central", peak=0.90,
            high_pct=25.0, mid_pct=15.0, mean_act=0.42, correct=True):
    return {
        "pred_class":          pred,
        "true_class":          pred if correct else "NV",
        "pred_confidence":     conf,
        "correct":             correct,
        "primary_region":      region,
        "peak_activation":     peak,
        "high_activation_pct": high_pct,
        "mid_activation_pct":  mid_pct,
        "mean_activation":     mean_act,
        "class_probs": {
            "BCC": 0.03, "BKL": 0.05, "MEL": 0.87, "NV": 0.05
        },
    }


# ==============================================================================
# TESTS — generate()
# ==============================================================================

class TestXAIReporterGenerate:

    def test_returns_required_keys(self):
        report = XAIReporter().generate(_sample())
        for key in [
            "xai_method", "target_layer", "prediction", "confidence",
            "attention_quality", "attention_note",
            "primary_focus_region", "region_clinical_note",
            "peak_activation", "high_activation_pct",
            "decision_margin", "margin_assessment",
            "top2_differential", "expected_morphological_features",
            "full_probability_distribution", "icd10_code", "disclaimer",
        ]:
            assert key in report, f"Missing key: {key}"

    def test_prediction_field_correct(self):
        report = XAIReporter().generate(_sample(pred="BCC"))
        assert report["prediction"] == "BCC"

    def test_confidence_formatted_as_percent(self):
        report = XAIReporter().generate(_sample(conf=0.87))
        assert "87" in report["confidence"] or "%" in report["confidence"]

    def test_strong_attention_high_peak(self):
        report = XAIReporter().generate(_sample(peak=0.92, high_pct=20.0))
        assert report["attention_quality"] == "Strong"

    def test_moderate_attention(self):
        report = XAIReporter().generate(_sample(peak=0.65, high_pct=8.0))
        assert report["attention_quality"] == "Moderate"

    def test_weak_attention_low_peak(self):
        report = XAIReporter().generate(_sample(peak=0.20, high_pct=1.0))
        assert report["attention_quality"] == "Weak / Diffuse"

    def test_high_margin_unambiguous(self):
        # conf=0.95 vs next class ~0.03 → margin ~0.92
        report = XAIReporter().generate(_sample(conf=0.95))
        assert "unambiguous" in report["margin_assessment"].lower() or \
               "very high" in report["margin_assessment"].lower()

    def test_low_margin_borderline(self):
        # Equal-ish probs → low margin
        sample = _sample(conf=0.28)
        sample["class_probs"] = {"BCC": 0.26, "BKL": 0.24, "MEL": 0.28, "NV": 0.22}
        report = XAIReporter().generate(sample)
        assert "borderline" in report["margin_assessment"].lower() or \
               "low" in report["margin_assessment"].lower()

    def test_expected_features_populated_for_known_class(self):
        for cls in ["BCC", "BKL", "MEL", "NV"]:
            sample = _sample(pred=cls)
            sample["class_probs"] = {cls: 0.90, "BCC": 0.05, "BKL": 0.03, "NV": 0.02}
            report = XAIReporter().generate(sample)
            assert len(report["expected_morphological_features"]) > 0

    def test_icd10_code_present_for_known_class(self):
        for cls in ["BCC", "BKL", "MEL", "NV"]:
            sample = _sample(pred=cls)
            sample["class_probs"] = {cls: 0.90, "BCC": 0.03, "BKL": 0.04, "NV": 0.03}
            report = XAIReporter().generate(sample)
            assert ICD10_CODES[cls][:3] in report["icd10_code"]

    def test_probability_distribution_all_classes(self):
        report = XAIReporter().generate(_sample())
        dist   = report["full_probability_distribution"]
        assert set(dist.keys()) == {"BCC", "BKL", "MEL", "NV"}

    def test_disclaimer_present(self):
        report = XAIReporter().generate(_sample())
        assert len(report["disclaimer"]) > 20

    def test_region_clinical_note_diffuse(self):
        report = XAIReporter().generate(_sample(region="diffuse"))
        assert "diffuse" in report["region_clinical_note"].lower() or \
               "global" in report["region_clinical_note"].lower()

    def test_repr(self):
        assert "XAIReporter" in repr(XAIReporter())


# ==============================================================================
# TESTS — build_medgemma_prompt()
# ==============================================================================

class TestBuildMedgemmaPrompt:

    def test_prompt_contains_pred_class(self):
        reporter = XAIReporter()
        sample   = _sample(pred="MEL")
        xai_rpt  = reporter.generate(sample)
        prompt   = reporter.build_medgemma_prompt(sample, xai_rpt)
        assert "MEL" in prompt

    def test_prompt_contains_confidence(self):
        reporter = XAIReporter()
        sample   = _sample(conf=0.87)
        xai_rpt  = reporter.generate(sample)
        prompt   = reporter.build_medgemma_prompt(sample, xai_rpt)
        assert "87" in prompt

    def test_prompt_contains_region(self):
        reporter = XAIReporter()
        sample   = _sample(region="central")
        xai_rpt  = reporter.generate(sample)
        prompt   = reporter.build_medgemma_prompt(sample, xai_rpt)
        assert "central" in prompt

    def test_prompt_contains_probabilities(self):
        reporter = XAIReporter()
        sample   = _sample()
        xai_rpt  = reporter.generate(sample)
        prompt   = reporter.build_medgemma_prompt(sample, xai_rpt)
        # All 4 class names should appear in probability table
        for cls in ["BCC", "BKL", "MEL", "NV"]:
            assert cls in prompt

    def test_prompt_is_non_empty_string(self):
        reporter = XAIReporter()
        sample   = _sample()
        xai_rpt  = reporter.generate(sample)
        prompt   = reporter.build_medgemma_prompt(sample, xai_rpt)
        assert isinstance(prompt, str) and len(prompt) > 100

    def test_medgemma_system_prompt_non_empty(self):
        assert isinstance(MEDGEMMA_SYSTEM_PROMPT, str)
        assert len(MEDGEMMA_SYSTEM_PROMPT) > 50
        assert "dermatol" in MEDGEMMA_SYSTEM_PROMPT.lower()


# ==============================================================================
# TESTS — domain knowledge tables
# ==============================================================================

class TestDomainKnowledgeTables:
    CLASSES = ["BCC", "BKL", "MEL", "NV"]

    def test_all_classes_in_descriptions(self):
        for cls in self.CLASSES:
            assert cls in CLASS_DESCRIPTIONS

    def test_all_classes_in_morphology(self):
        for cls in self.CLASSES:
            assert cls in CLASS_MORPHOLOGY
            assert len(CLASS_MORPHOLOGY[cls]) > 20

    def test_all_classes_in_expected_features(self):
        for cls in self.CLASSES:
            assert cls in EXPECTED_FEATURES
            assert len(EXPECTED_FEATURES[cls]) >= 3

    def test_all_classes_in_risk_levels(self):
        for cls in self.CLASSES:
            assert cls in RISK_LEVELS
            assert RISK_LEVELS[cls] in ("Low", "High", "Critical")

    def test_mel_is_critical_risk(self):
        assert RISK_LEVELS["MEL"] == "Critical"

    def test_nv_and_bkl_are_low_risk(self):
        assert RISK_LEVELS["NV"]  == "Low"
        assert RISK_LEVELS["BKL"] == "Low"

    def test_all_classes_have_icd10(self):
        for cls in self.CLASSES:
            assert cls in ICD10_CODES
            assert len(ICD10_CODES[cls]) > 2

    def test_mel_icd10_starts_with_c43(self):
        assert ICD10_CODES["MEL"].startswith("C43")

    def test_all_classes_have_next_steps(self):
        for cls in self.CLASSES:
            assert cls in NEXT_STEPS
            assert len(NEXT_STEPS[cls]) > 20

    def test_mel_next_step_is_urgent(self):
        assert "URGENT" in NEXT_STEPS["MEL"].upper() or \
               "urgent" in NEXT_STEPS["MEL"]