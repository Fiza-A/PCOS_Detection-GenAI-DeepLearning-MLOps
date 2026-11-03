"""Final held-out evaluation tests; no training, tuning, or report rewriting."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DEEP_LEARNING" / "src"))
import evaluate as evaluation  # noqa: E402


class FinalTestEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.checkpoint = evaluation.load_frozen_checkpoint()
        cls.X_test, cls.y_test = evaluation.load_untouched_test_data()
        cls.probabilities = evaluation.predict_probabilities(cls.model, cls.X_test)
        cls.repeat_probabilities = evaluation.predict_probabilities(cls.model, cls.X_test)
        cls.computed = evaluation.calculate_test_metrics(cls.y_test, cls.probabilities)
        cls.saved_result = json.loads((evaluation.REPORT_DIR / "test_evaluation.json").read_text(encoding="utf-8"))

    def test_01_checkpoint_exists(self) -> None:
        self.assertTrue(evaluation.CHECKPOINT_PATH.is_file())

    def test_02_checkpoint_loads_with_expected_architecture(self) -> None:
        self.assertIsInstance(self.model, evaluation.PCOSDLNN)
        self.assertEqual(self.model.config, evaluation.ModelConfig())
        self.assertEqual(self.checkpoint["experiment"], "REAL_ONLY_BASELINE")

    def test_03_test_shape_is_82_by_58(self) -> None:
        self.assertEqual(self.X_test.shape, (82, 58))

    def test_04_test_labels_have_expected_classes_and_counts(self) -> None:
        self.assertEqual(set(np.unique(self.y_test)), {0, 1})
        self.assertEqual((int((self.y_test == 0).sum()), int((self.y_test == 1).sum())), (55, 27))

    def test_05_prediction_count_matches_test_size(self) -> None:
        self.assertEqual(len(self.probabilities), len(self.y_test))

    def test_06_probabilities_are_finite(self) -> None:
        self.assertTrue(np.isfinite(self.probabilities).all())

    def test_07_probabilities_are_bounded(self) -> None:
        self.assertTrue(((self.probabilities >= 0.0) & (self.probabilities <= 1.0)).all())

    def test_08_predicted_labels_are_binary(self) -> None:
        self.assertEqual(set(np.unique(self.computed["predicted_labels"])), {0, 1})

    def test_09_confusion_counts_sum_to_test_size(self) -> None:
        counts = self.computed["confusion_matrix"]
        self.assertEqual(sum(counts.values()), 82)

    def test_10_metrics_are_finite(self) -> None:
        self.assertTrue(all(np.isfinite(value) for value in self.computed["metrics"].values()))

    def test_11_roc_auc_is_in_unit_interval(self) -> None:
        value = self.computed["metrics"]["roc_auc"]
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_12_pr_auc_is_in_unit_interval(self) -> None:
        value = self.computed["metrics"]["pr_auc"]
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_13_primary_threshold_is_exactly_point_five(self) -> None:
        self.assertEqual(evaluation.THRESHOLD, 0.50)
        self.assertEqual(self.saved_result["threshold"], 0.50)
        self.assertFalse(self.saved_result["test_arrays_used_for_threshold_selection"])

    def test_14_saved_predictions_have_82_rows(self) -> None:
        predictions = pd.read_csv(evaluation.PREDICTIONS_CSV)
        self.assertEqual(len(predictions), 82)
        self.assertEqual(list(predictions.columns), ["test_index", "true_label", "predicted_probability", "predicted_label"])

    def test_15_inference_is_deterministic(self) -> None:
        np.testing.assert_array_equal(self.probabilities, self.repeat_probabilities)
        repeated_metrics = evaluation.calculate_test_metrics(self.y_test, self.repeat_probabilities)
        self.assertEqual(self.computed["metrics"], repeated_metrics["metrics"])
        self.assertEqual(self.computed["confusion_matrix"], repeated_metrics["confusion_matrix"])
        self.assertTrue(self.saved_result["test_predictions_deterministic_on_repeat"])

    def test_16_model_is_in_eval_mode(self) -> None:
        self.assertFalse(self.model.training)
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d)):
                self.assertFalse(module.training)

    def test_17_confusion_matrix_artifact_matches_computed_counts(self) -> None:
        saved = pd.read_csv(evaluation.CONFUSION_CSV).iloc[0].to_dict()
        self.assertEqual({key: int(value) for key, value in saved.items()}, self.computed["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
