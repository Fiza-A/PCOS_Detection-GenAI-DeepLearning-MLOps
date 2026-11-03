"""Phase 3.3 utility tests; tests do not invoke the training loop."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DEEP_LEARNING" / "src"))
from model import ModelConfig, PCOSDLNN, initialize_model  # noqa: E402
from train import (  # noqa: E402
    HISTORY_COLUMNS,
    TrainingConfig,
    calculate_positive_class_weight,
    load_checkpoint,
    load_training_validation_data,
    save_checkpoint,
    train_model,
    validate_history_schema,
)
import pandas as pd  # noqa: E402


class TrainingUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X_train, cls.y_train, cls.X_validation, cls.y_validation, cls.report = load_training_validation_data()

    def test_01_training_and_validation_data_shapes(self) -> None:
        self.assertEqual(self.X_train.shape, (378, 58))
        self.assertEqual(self.y_train.shape, (378,))
        self.assertEqual(self.X_validation.shape, (81, 58))
        self.assertEqual(self.y_validation.shape, (81,))

    def test_02_class_weight_uses_training_label_counts(self) -> None:
        weight, negatives, positives = calculate_positive_class_weight(self.y_train)
        self.assertEqual((negatives, positives), (254, 124))
        self.assertAlmostEqual(weight, 254 / 124)

    def test_03_class_weight_is_finite_positive(self) -> None:
        weight, _, _ = calculate_positive_class_weight(self.y_train)
        self.assertTrue(np.isfinite(weight))
        self.assertGreater(weight, 0.0)

    def test_04_training_configuration_matches_phase_specification(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.patience, 15)

    def test_05_model_output_shape(self) -> None:
        model = initialize_model(seed=42, config=ModelConfig(input_dim=58))
        self.assertEqual(tuple(model(torch.zeros((4, 58))).shape), (4, 1))

    def test_06_weighted_bce_with_logits_accepts_output(self) -> None:
        weight, _, _ = calculate_positive_class_weight(self.y_train)
        model = initialize_model(seed=42)
        logits = model(torch.zeros((4, 58)))
        targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())

    def test_07_checkpoint_save_and_load(self) -> None:
        model = initialize_model(seed=42)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        config = TrainingConfig()
        with tempfile.TemporaryDirectory(prefix=".training_checkpoint_test_", dir=ROOT / "DEEP_LEARNING") as folder:
            checkpoint = Path(folder) / "checkpoint.pt"
            save_checkpoint(checkpoint, model, optimizer, config, 1, 0.5, 254 / 124, 254, 124)
            restored, payload = load_checkpoint(checkpoint)
        self.assertEqual(payload["epoch"], 1)
        self.assertEqual(payload["training_class_counts"], {"negative": 254, "positive": 124})
        for left, right in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_08_training_history_schema(self) -> None:
        history = pd.DataFrame(
            [{"epoch": 1, "training_loss": 0.8, "validation_loss": 0.9, "learning_rate": 0.001}],
            columns=HISTORY_COLUMNS,
        )
        validate_history_schema(history)
        self.assertEqual(tuple(history.columns), HISTORY_COLUMNS)

    def test_09_training_function_has_no_test_data_argument(self) -> None:
        self.assertNotIn("X_test", inspect.signature(train_model).parameters)
        self.assertNotIn("y_test", inspect.signature(train_model).parameters)
        self.assertFalse(self.report.get("synthetic_data_used", True))

    def test_10_test_arrays_are_not_loaded_by_data_loader(self) -> None:
        source = inspect.getsource(load_training_validation_data)
        self.assertNotIn('"X_test.npy"', source)
        self.assertNotIn('"y_test.npy"', source)
        self.assertEqual(self.report["matrix_shapes"]["test"], [82, 58])


if __name__ == "__main__":
    unittest.main()
