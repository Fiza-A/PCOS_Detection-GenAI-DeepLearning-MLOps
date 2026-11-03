"""Architecture-only tests. No optimizer step, fitting, or model metrics occur."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DEEP_LEARNING" / "src"))
from model import ModelConfig, PCOSDLNN, initialize_model, logits_to_probabilities  # noqa: E402


class PCOSDLNNArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = initialize_model(seed=42)

    def test_01_model_instantiates(self) -> None:
        self.assertIsInstance(self.model, nn.Module)

    def test_02_input_dimension_is_58(self) -> None:
        report = json.loads((ROOT / "DEEP_LEARNING/reports/preprocessing_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["n_features_after_encoding"], 58)
        self.assertEqual(self.model.config.input_dim, 58)

    def test_03_hidden_dimensions_are_128_64_32(self) -> None:
        self.assertEqual((self.model.fc1.out_features, self.model.fc2.out_features, self.model.fc3.out_features), (128, 64, 32))

    def test_04_dropout_rates_are_configured(self) -> None:
        self.assertAlmostEqual(self.model.dropout1.p, 0.30)
        self.assertAlmostEqual(self.model.dropout2.p, 0.20)

    def test_05_output_dimension_is_one(self) -> None:
        self.assertEqual(self.model.fc_out.out_features, 1)
        self.assertEqual(self.model.config.output_dim, 1)

    def test_06_forward_accepts_batch_by_58_input(self) -> None:
        result = self.model(torch.randn(4, 58))
        self.assertEqual(tuple(result.shape), (4, 1))

    def test_07_forward_output_shape(self) -> None:
        result = self.model(torch.randn(7, self.model.config.input_dim))
        self.assertEqual(tuple(result.shape), (7, 1))

    def test_08_forward_output_is_finite(self) -> None:
        result = self.model(torch.randn(4, 58))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_09_bce_with_logits_accepts_output(self) -> None:
        logits = self.model(torch.randn(4, 58))
        targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_10_sigmoid_probability_conversion_is_bounded(self) -> None:
        logits = torch.tensor([[-100.0], [0.0], [100.0]])
        probabilities = logits_to_probabilities(logits)
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all().item())

    def test_11_same_seed_reproduces_initial_parameters(self) -> None:
        first = initialize_model(seed=42)
        second = initialize_model(seed=42)
        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter, rtol=0.0, atol=0.0)

    def test_12_layer_structure_and_parameter_count(self) -> None:
        self.assertEqual(
            list(self.model._modules),
            ["fc1", "relu1", "bn1", "dropout1", "fc2", "relu2", "bn2", "dropout2", "fc3", "relu3", "fc_out"],
        )
        self.assertEqual(self.model.fc1.weight.shape, (128, 58))
        self.assertEqual(self.model.fc2.weight.shape, (64, 128))
        self.assertEqual(self.model.fc3.weight.shape, (32, 64))
        self.assertEqual(self.model.fc_out.weight.shape, (1, 32))
        self.assertEqual(sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad), 18305)

    def test_config_is_the_single_architecture_source(self) -> None:
        config = ModelConfig()
        self.assertEqual((config.input_dim, config.hidden_1, config.hidden_2, config.hidden_3, config.output_dim), (58, 128, 64, 32, 1))
        self.assertEqual((config.dropout_1, config.dropout_2), (0.30, 0.20))

    def test_model_rejects_wrong_feature_width(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.randn(4, 57))


if __name__ == "__main__":
    unittest.main()
