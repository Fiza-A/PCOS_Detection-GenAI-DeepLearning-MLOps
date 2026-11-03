"""MLOps quality, registry, deployment, privacy logging, and monitoring tests."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MLOPS" / "src"))

from common import write_json  # noqa: E402
from deployment import check_deployment_readiness  # noqa: E402
from monitoring import aggregate_monitoring, registry_model_state  # noqa: E402
from pipeline import _load_config  # noqa: E402
from prediction_logging import log_prediction  # noqa: E402
from quality_gate import CONFIG_KEYS, evaluate_quality_gate  # noqa: E402
from registry import register_if_eligible  # noqa: E402


class MLOpsPhase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = json.loads((ROOT / "DEEP_LEARNING" / "reports" / "test_evaluation.json").read_text(encoding="utf-8"))["metrics"]
        cls.model_metadata = {
            "model_name": "test candidate",
            "training_run_id": "test-run",
            "dataset_version": "test-dataset",
            "artifact_path": "MLOPS/tests/model.bin",
            "artifact_sha256": "unit-test-hash",
            "test_metrics": dict(cls.metrics),
            "registered_at_utc": "test-only-timestamp",
        }

    def test_quality_gate_missing_thresholds_are_insufficient_configuration(self) -> None:
        empty_thresholds = {key: None for key in CONFIG_KEYS.values()}
        result = evaluate_quality_gate(self.metrics, empty_thresholds)
        self.assertEqual(result["status"], "INSUFFICIENT_CONFIGURATION")
        self.assertFalse(result["eligible_for_registration"])
        self.assertEqual(len(result["reasons"]), 5)
        self.assertTrue(all(item["status"] == "INSUFFICIENT_CONFIGURATION" for item in result["checks"].values()))

    def test_quality_gate_passes_when_all_configured_minima_are_met(self) -> None:
        thresholds = {key: self.metrics[metric] for metric, key in CONFIG_KEYS.items()}
        result = evaluate_quality_gate(self.metrics, thresholds)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["eligible_for_registration"])
        for metric, item in result["checks"].items():
            self.assertEqual(item["observed"], self.metrics[metric])
            self.assertEqual(item["status"], "PASS")

    def test_quality_gate_fails_when_observed_metrics_miss_configured_minimums(self) -> None:
        thresholds = {key: 1.0 for key in CONFIG_KEYS.values()}
        result = evaluate_quality_gate(self.metrics, thresholds)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["eligible_for_registration"])
        self.assertTrue(all(item["status"] == "FAIL" for item in result["checks"].values()))

    def test_missing_observed_metric_fails(self) -> None:
        thresholds = {key: 0.0 for key in CONFIG_KEYS.values()}
        metrics = dict(self.metrics)
        metrics.pop("roc_auc")
        result = evaluate_quality_gate(metrics, thresholds)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["checks"]["roc_auc"]["status"], "FAIL")

    def test_failed_or_insufficient_gate_tracks_but_does_not_register_model(self) -> None:
        for gate_status in ("FAIL", "INSUFFICIENT_CONFIGURATION"):
            with self.subTest(gate_status=gate_status), tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
                registry_path = Path(folder) / "registry.json"
                with patch("registry.REGISTRY_PATH", registry_path):
                    result = register_if_eligible("test-version", self.model_metadata, {"eligible_for_registration": False, "status": gate_status})
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "NOT_REGISTERED")
                self.assertFalse(result["production_model_changed"])
                self.assertEqual(registry["registered_models"], [])
                self.assertEqual(registry["tracked_models"][0]["lifecycle_status"], "TRAINED")
                self.assertIsNone(registry["production_model"])

    def test_passed_gate_registers_candidate_without_deployment_or_promotion(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            registry_path = Path(folder) / "registry.json"
            with patch("registry.REGISTRY_PATH", registry_path):
                result = register_if_eligible("test-version", self.model_metadata, {"eligible_for_registration": True, "status": "PASS"})
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "REGISTERED")
            self.assertFalse(result["production_model_changed"])
            self.assertEqual(registry["registered_models"][0]["lifecycle_status"], "REGISTERED")
            self.assertEqual(registry["production_model"], None)

    def _readiness_fixture(self, folder: Path, gate_status: str = "PASS") -> dict[str, object]:
        model = folder / "model.bin"
        preprocessing = folder / "preprocessing.bin"
        evaluation = folder / "evaluation.json"
        metadata = folder / "version.json"
        registry = folder / "registry.json"
        model.write_bytes(b"test-model")
        preprocessing.write_bytes(b"test-preprocessing")
        write_json(evaluation, {"metrics": self.metrics})
        model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
        prep_hash = hashlib.sha256(preprocessing.read_bytes()).hexdigest()
        write_json(metadata, {
            "model_version": "candidate-v1",
            "artifact_path": str(model),
            "artifact_sha256": model_hash,
            "training_run_id": "run-v1",
            "dataset_version": "data-v1",
        })
        registered_models = []
        if gate_status == "PASS":
            registered_models.append({"model_version": "candidate-v1", "artifact_sha256": model_hash})
        write_json(registry, {"registered_models": registered_models, "production_model": None})
        readiness = check_deployment_readiness(
            model_path=model,
            preprocessing_path=preprocessing,
            evaluation_path=evaluation,
            version_metadata_path=metadata,
            registry_path=registry,
            expected_model_sha256=model_hash,
            expected_preprocessing_sha256=prep_hash,
            quality_gate={"status": gate_status},
            model_version="candidate-v1",
        )
        return {"readiness": readiness, "model": model, "preprocessing": preprocessing, "evaluation": evaluation, "metadata": metadata, "registry": registry}

    def test_deployment_readiness_when_gate_passes_and_artifacts_are_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            result = self._readiness_fixture(Path(folder))
            self.assertTrue(result["readiness"]["deployment_ready"])
            self.assertEqual(result["readiness"]["status"], "READY")
            self.assertFalse(result["readiness"]["deployment_performed"])

    def test_deployment_blocked_when_quality_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            result = self._readiness_fixture(Path(folder), gate_status="FAIL")
            self.assertFalse(result["readiness"]["deployment_ready"])
            self.assertFalse(result["readiness"]["checks"]["quality_gate_passes"]["passed"])

    def test_deployment_readiness_reports_missing_or_invalid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            base = self._readiness_fixture(Path(folder))
            result = check_deployment_readiness(
                model_path=Path(folder) / "missing-model.pt",
                preprocessing_path=base["preprocessing"],
                evaluation_path=Path(folder) / "missing-evaluation.json",
                version_metadata_path=Path(folder) / "missing-version.json",
                registry_path=Path(folder) / "missing-registry.json",
                expected_model_sha256="wrong",
                expected_preprocessing_sha256="wrong",
                quality_gate={"status": "PASS"},
                model_version="candidate-v1",
            )
            self.assertFalse(result["deployment_ready"])
            self.assertIn("model_artifact_exists", result["checks"])
            self.assertFalse(result["checks"]["evaluation_metrics_exist"]["passed"])

    def test_prediction_logging_contains_no_raw_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            log_path = Path(folder) / "predictions.jsonl"
            with patch("prediction_logging.PREDICTION_LOG", log_path):
                event = log_prediction(model_version="model-v1", status="SUCCESS", inference_duration_ms=4.25, predicted_label=1, probability=0.74)
                failure = log_prediction(model_version="model-v1", status="FAILURE", inference_duration_ms=1.0, error_category="input_validation")
            text = log_path.read_text(encoding="utf-8")
            self.assertEqual(event["status"], "SUCCESS")
            self.assertEqual(failure["status"], "FAILURE")
            self.assertNotIn("BMI", text)
            self.assertNotIn("Age", text)
            self.assertNotIn("raw_input", text)
            self.assertNotIn("patient_id", text)
            self.assertNotIn("clinical_values", text)

    def test_monitoring_aggregation_counts_and_distribution(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            log_path = Path(folder) / "predictions.jsonl"
            with patch("prediction_logging.PREDICTION_LOG", log_path):
                log_prediction(model_version="model-v1", status="SUCCESS", inference_duration_ms=4.0, predicted_label=1, probability=0.7)
                log_prediction(model_version="model-v1", status="SUCCESS", inference_duration_ms=8.0, predicted_label=0, probability=0.2)
                log_prediction(model_version="model-v1", status="FAILURE", inference_duration_ms=2.0, error_category="input_validation")
            result = aggregate_monitoring(
                log_path,
                report_path=None,
                candidate_model_version="model-v1",
                serving_model_version="model-v1",
                deployment_status="DEPLOYED",
            )
            self.assertEqual(result["total_predictions"], 3)
            self.assertEqual(result["successful_predictions"], 2)
            self.assertEqual(result["failed_predictions"], 1)
            self.assertAlmostEqual(result["success_rate"], 2 / 3)
            self.assertEqual(result["prediction_distribution"], {"negative": 1, "positive": 1})
            self.assertEqual(result["average_inference_latency_ms"], 6.0)
            self.assertEqual(result["candidate_model_version"], "model-v1")
            self.assertEqual(result["serving_model_version"], "model-v1")
            self.assertEqual(result["deployment_status"], "DEPLOYED")
            self.assertIsNone(result["production_performance_metrics"])
            self.assertIn("ground-truth labels are not available", result["production_performance_note"])

    def test_candidate_is_not_reported_as_serving_without_deployment(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            log_path = Path(folder) / "predictions.jsonl"
            with patch("prediction_logging.PREDICTION_LOG", log_path):
                log_prediction(model_version="candidate-v1", status="SUCCESS", inference_duration_ms=4.0, predicted_label=1, probability=0.7)
            result = aggregate_monitoring(
                log_path,
                report_path=None,
                candidate_model_version="candidate-v1",
                serving_model_version=None,
                deployment_status="NOT_DEPLOYED",
            )
            self.assertEqual(result["candidate_model_version"], "candidate-v1")
            self.assertIsNone(result["serving_model_version"])
            self.assertEqual(result["deployment_status"], "NOT_DEPLOYED")

    def test_registry_without_production_pointer_has_no_serving_version(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            registry_path = Path(folder) / "registry.json"
            registry_path.write_text(json.dumps({
                "production_model": None,
                "tracked_models": [{"model_version": "candidate-v1"}],
                "registered_models": [],
            }), encoding="utf-8")
            state = registry_model_state(registry_path)
            self.assertEqual(state["candidate_model_version"], "candidate-v1")
            self.assertIsNone(state["serving_model_version"])
            self.assertEqual(state["deployment_status"], "NOT_DEPLOYED")

    def test_drift_readiness_is_insufficient_without_production_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "MLOPS" / "tests") as folder:
            result = aggregate_monitoring(Path(folder) / "no-events.jsonl", report_path=None)
            for item in result["drift_readiness"].values():
                self.assertEqual(item["status"], "INSUFFICIENT_DATA")
            self.assertEqual(result["total_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
