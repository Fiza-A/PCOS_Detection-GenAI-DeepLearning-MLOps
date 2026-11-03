"""Proposed PCOS DLNN architecture; this module does not train or predict."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Single source of architecture dimensions and dropout settings."""

    input_dim: int = 58
    hidden_1: int = 128
    hidden_2: int = 64
    hidden_3: int = 32
    dropout_1: float = 0.30
    dropout_2: float = 0.20
    output_dim: int = 1


class PCOSDLNN(nn.Module):
    """Dense binary classifier that returns one unnormalized logit per row."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if min(cfg.input_dim, cfg.hidden_1, cfg.hidden_2, cfg.hidden_3, cfg.output_dim) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if cfg.output_dim != 1:
            raise ValueError("Binary PCOS classification requires output_dim=1.")
        if not (0.0 <= cfg.dropout_1 < 1.0 and 0.0 <= cfg.dropout_2 < 1.0):
            raise ValueError("Dropout probabilities must be in [0, 1).")

        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(cfg.hidden_1)
        self.dropout1 = nn.Dropout(cfg.dropout_1)

        self.fc2 = nn.Linear(cfg.hidden_1, cfg.hidden_2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(cfg.hidden_2)
        self.dropout2 = nn.Dropout(cfg.dropout_2)

        self.fc3 = nn.Linear(cfg.hidden_2, cfg.hidden_3)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(cfg.hidden_3, cfg.output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return a shape ``(batch, 1)`` logit tensor (without sigmoid)."""
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected input shaped (batch, {self.config.input_dim}); got {tuple(inputs.shape)}."
            )
        values = self.dropout1(self.bn1(self.relu1(self.fc1(inputs))))
        values = self.dropout2(self.bn2(self.relu2(self.fc2(values))))
        values = self.relu3(self.fc3(values))
        return self.fc_out(values)


def initialize_model(seed: int = 42, config: ModelConfig | None = None) -> PCOSDLNN:
    """Seed PyTorch and initialize a model reproducibly, without training it."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return PCOSDLNN(config=config)


def logits_to_probabilities(logits: Tensor) -> Tensor:
    """Convert model logits to binary probabilities at the inference boundary."""
    return torch.sigmoid(logits)
