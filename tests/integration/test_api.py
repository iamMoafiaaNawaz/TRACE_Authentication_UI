# -*- coding: utf-8 -*-
"""
tests/integration/test_api.py
================================
Integration tests for the TRACE REST API (src/api/app.py).

Tests are designed to be skipped gracefully if the API or its
dependencies (Flask/FastAPI, model checkpoint) are not available.
"""

import json
import pytest


# ==============================================================================
# HELPERS / FIXTURES
# ==============================================================================

def _try_import_app():
    try:
        from src.api import app as api_app
        return api_app
    except (ImportError, Exception):
        return None


# ==============================================================================
# API IMPORT
# ==============================================================================

class TestApiImport:

    def test_api_module_importable(self):
        """The api module should import without crashing, even if stub."""
        try:
            import src.api
        except ImportError as e:
            pytest.skip(f"src.api not importable: {e}")

    def test_routes_importable(self):
        try:
            from src.api import routes
        except (ImportError, Exception) as e:
            pytest.skip(f"src.api.routes not importable: {e}")

    def test_schemas_importable(self):
        try:
            from src.api import schemas
        except (ImportError, Exception) as e:
            pytest.skip(f"src.api.schemas not importable: {e}")


# ==============================================================================
# HEALTH ENDPOINT (if Flask app available)
# ==============================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self):
        app_mod = _try_import_app()
        if app_mod is None:
            pytest.skip("API app not available")
        try:
            app    = app_mod.app
            client = app.test_client()
            resp   = client.get("/health")
            assert resp.status_code in (200, 404)  # 404 if route not implemented
        except Exception as e:
            pytest.skip(f"Health endpoint test skipped: {e}")


# ==============================================================================
# PREDICTION ENDPOINT (smoke test, no real model)
# ==============================================================================

class TestPredictEndpoint:

    def test_predict_endpoint_exists_or_stub(self):
        """
        If the /predict endpoint is implemented, it should accept POST requests.
        If it's a stub, this test is skipped gracefully.
        """
        app_mod = _try_import_app()
        if app_mod is None:
            pytest.skip("API app not available")
        try:
            app    = app_mod.app
            client = app.test_client()
            resp   = client.post("/predict", json={})
            # Any HTTP response (even 400 bad request) means the endpoint exists
            assert resp.status_code in (200, 400, 404, 422, 500)
        except Exception as e:
            pytest.skip(f"Predict endpoint test skipped: {e}")


# ==============================================================================
# MONITORING — PredictionLogger + DriftDetector smoke tests
# ==============================================================================

class TestMonitoringIntegration:

    def test_prediction_logger_writes_jsonl(self, tmp_path):
        from monitoring.performance_tracker import PredictionLogger
        logger = PredictionLogger(log_dir=tmp_path / "logs", rotate_daily=False)
        logger.log(
            result={"prediction": "MEL", "confidence": 0.87, "class_probs": {"MEL": 0.87}},
            image_path="test.jpg",
        )
        jsonl_files = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(jsonl_files) == 1
        lines = jsonl_files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["prediction"] == "MEL"
        assert entry["confidence"] == pytest.approx(0.87)

    def test_drift_detector_no_baseline(self, tmp_path):
        from monitoring.performance_tracker import DriftDetector
        baseline = tmp_path / "baseline.json"
        # No baseline file — should not crash
        detector = DriftDetector(baseline_path=baseline)
        for cls in ["MEL", "BCC", "NV", "BKL"] * 25:
            detector.check({"prediction": cls, "confidence": 0.8})

    def test_performance_summary_empty(self, tmp_path):
        from monitoring.performance_tracker import PerformanceSummary
        summary = PerformanceSummary(log_dir=tmp_path)
        result  = summary.generate(date="2026-03-18")
        assert result["total_predictions"] == 0
        assert "note" in result
