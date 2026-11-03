"""SHAP artifact, prediction, and source-integrity verification tests."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DEEP_LEARNING" / "src"))
import explain as xai  # noqa: E402
from evaluate import load_frozen_checkpoint, predict_probabilities  # noqa: E402


class ExplainabilityArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = json.loads(xai.METADATA_PATH.read_text(encoding="utf-8"))
        cls.shap_values = np.load(xai.SHAP_VALUES_PATH, allow_pickle=False)
        cls.feature_names = json.loads(xai.FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        cls.importance = pd.read_csv(xai.IMPORTANCE_CSV)
        cls.local = pd.read_csv(xai.LOCAL_CSV)
        cls.X_test = np.load(xai.X_TEST_PATH, allow_pickle=False)
        cls.y_test = np.load(xai.Y_TEST_PATH, allow_pickle=False)
        cls.predictions = pd.read_csv(xai.PREDICTIONS_CSV)

    def test_01_shap_values_exist(self) -> None:
        self.assertTrue(xai.SHAP_VALUES_PATH.is_file())

    def test_02_shap_dimensions_are_82_by_58(self) -> None:
        self.assertEqual(self.shap_values.shape, (82, 58))

    def test_03_all_test_samples_are_explained(self) -> None:
        self.assertEqual(self.metadata["explained_samples"], 82)
        self.assertEqual(len(self.shap_values), 82)

    def test_04_authoritative_feature_names_count(self) -> None:
        self.assertEqual(self.metadata["feature_count"], 58)
        self.assertEqual(len(self.feature_names), 58)
        self.assertEqual(len(set(self.feature_names)), 58)

    def test_05_shap_values_are_finite(self) -> None:
        self.assertTrue(np.isfinite(self.shap_values).all())

    def test_06_global_importance_has_all_features(self) -> None:
        self.assertEqual(len(self.importance), 58)
        self.assertEqual(set(self.importance["feature"]), set(self.feature_names))
        self.assertEqual(list(self.importance.columns), ["rank", "feature", "mean_abs_shap"])

    def test_07_feature_ranking_matches_shap_values_deterministically(self) -> None:
        expected_order = np.argsort(-np.mean(np.abs(self.shap_values), axis=0), kind="stable")
        self.assertEqual(self.importance["feature"].tolist(), [self.feature_names[i] for i in expected_order])
        self.assertEqual(self.importance["rank"].tolist(), list(range(1, 59)))
        self.assertTrue(self.importance["mean_abs_shap"].is_monotonic_decreasing)

    def test_08_predictions_match_phase_34(self) -> None:
        model, _ = load_frozen_checkpoint()
        actual = predict_probabilities(model, self.X_test)
        saved = self.predictions["predicted_probability"].to_numpy(dtype=float)
        self.assertTrue(np.allclose(actual, saved, rtol=0.0, atol=xai.PREDICTION_CSV_SERIALIZATION_ATOL))
        self.assertTrue(np.array_equal((actual >= 0.50).astype(int), self.predictions["predicted_label"].to_numpy(dtype=int)))
        self.assertTrue(self.metadata["test_prediction_consistency_with_phase_3_4"])

    def test_09_model_checkpoint_hash_is_unchanged(self) -> None:
        expected = self.metadata["integrity_hashes_sha256"]["checkpoint"]
        self.assertEqual(xai.file_sha256(xai.CHECKPOINT_PATH), expected)
        self.assertFalse(self.metadata["checkpoint_modified"])

    def test_10_test_arrays_hashes_are_unchanged(self) -> None:
        hashes = self.metadata["integrity_hashes_sha256"]
        self.assertEqual(xai.file_sha256(xai.X_TEST_PATH), hashes["X_test"])
        self.assertEqual(xai.file_sha256(xai.Y_TEST_PATH), hashes["y_test"])
        self.assertFalse(self.metadata["test_data_modified"])

    def test_11_no_synthetic_data_or_generator_was_used(self) -> None:
        self.assertFalse(self.metadata["synthetic_data_used"])
        self.assertFalse(self.metadata["ctgan_or_tvae_loaded"])

    def test_12_no_training_occurred(self) -> None:
        self.assertFalse(self.metadata["training_performed"])

    def test_13_background_is_deterministic_training_only_subset(self) -> None:
        self.assertEqual(self.metadata["background_dataset"], "DEEP_LEARNING\\data\\processed\\X_train.npy")
        expected = np.sort(np.random.RandomState(42).choice(378, 100, replace=False)).tolist()
        self.assertEqual(self.metadata["background_training_row_positions"], expected)
        self.assertEqual(self.metadata["background_size"], 100)
        self.assertTrue(all(0 <= row < 378 for row in expected))

    def test_14_individual_case_selection_is_deterministic(self) -> None:
        predicted = self.predictions["predicted_label"].to_numpy(dtype=int)
        expected = xai._select_individual_cases(self.y_test, predicted)
        self.assertEqual(self.metadata["individual_cases"], expected)
        self.assertEqual(set(expected), {"true_negative", "true_positive", "false_positive", "false_negative"})

    def test_15_repeated_explanation_is_reproducible(self) -> None:
        self.assertTrue(self.metadata["repeated_shap_values_within_tolerance"])
        self.assertLessEqual(self.metadata["shap_repeat_max_absolute_delta"], self.metadata["reproducibility_absolute_tolerance"])

    def test_16_additivity_diagnostic_is_measured_and_documented(self) -> None:
        check = self.metadata["additivity_check"]
        self.assertEqual(check["status"], "APPROXIMATE_DIAGNOSTIC_ONLY")
        self.assertTrue(np.isfinite(check["max_absolute_residual"]))
        self.assertTrue(np.isfinite(check["mean_absolute_residual"]))
        self.assertIn("P(PCOS = 1)", self.metadata["output_space"])

    def test_17_local_rows_are_top_ten_for_each_available_case(self) -> None:
        for case_type, test_index in self.metadata["individual_cases"].items():
            if test_index is None:
                continue
            case = self.local[self.local["case_type"] == case_type]
            self.assertEqual(len(case), 10)
            self.assertTrue(case["abs_shap_value"].is_monotonic_decreasing)
            self.assertTrue((case["test_index"] == test_index).all())
            self.assertTrue((case["predicted_probability"] >= 0.0).all())
            self.assertTrue((case["predicted_probability"] <= 1.0).all())

    def test_18_required_visualizations_exist(self) -> None:
        self.assertTrue((xai.REPORT_DIR / "shap_summary.png").is_file())
        self.assertTrue((xai.REPORT_DIR / "shap_global_importance.png").is_file())
        for case_type, plot_path in self.metadata["individual_case_plots"].items():
            if self.metadata["individual_cases"][case_type] is not None:
                self.assertTrue((ROOT / plot_path).is_file())

    def test_19_all_input_artifact_hashes_match(self) -> None:
        self.assertTrue(self.metadata["artifact_integrity_passed"])
        for name, path in xai.INTEGRITY_PATHS.items():
            self.assertEqual(xai.file_sha256(path), self.metadata["integrity_hashes_sha256"][name])


if __name__ == "__main__":
    unittest.main()
