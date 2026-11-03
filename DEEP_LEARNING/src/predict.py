"""Inference-only adapter for the frozen Phase 3 DLNN and fitted preprocessing."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DL_ROOT = ROOT / "DEEP_LEARNING"
MODEL_PATH = DL_ROOT / "models" / "dl_baseline_best.pt"
PREPROCESSOR_PATH = DL_ROOT / "models" / "preprocessing_pipeline.joblib"
PREPROCESSING_REPORT_PATH = DL_ROOT / "reports" / "preprocessing_report.json"
FEATURE_NAMES_PATH = DL_ROOT / "reports" / "transformed_feature_names.json"
EVALUATION_PATH = DL_ROOT / "reports" / "test_evaluation.json"
SRC = DL_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import load_frozen_checkpoint, predict_probabilities  # noqa: E402


class PCOSPredictor:
    """Loads only frozen inference artifacts; never fits or writes artifacts."""

    def __init__(self) -> None:
        required = (MODEL_PATH, PREPROCESSOR_PATH, PREPROCESSING_REPORT_PATH, FEATURE_NAMES_PATH, EVALUATION_PATH)
        missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Dashboard prerequisite artifacts missing: " + ", ".join(missing))

        self.report: dict[str, Any] = json.loads(PREPROCESSING_REPORT_PATH.read_text(encoding="utf-8"))
        self.feature_names: list[str] = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
        evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        self.threshold = float(evaluation["threshold"])
        if self.threshold != 0.50:
            raise ValueError(f"Official threshold artifact is {self.threshold}; expected the established 0.50.")
        preprocessing = self.report["preprocessing"]
        self.numerical_features = list(preprocessing["numerical_features"])
        self.categorical_features = list(preprocessing["categorical_features"])
        self.input_features = self.numerical_features + self.categorical_features
        if len(self.feature_names) != 58 or len(self.input_features) != 41:
            raise ValueError("Unexpected model schema; expected 41 source features and 58 transformed features.")
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        fitted_names = list(self.preprocessor.get_feature_names_out())
        if fitted_names != self.feature_names:
            raise ValueError("Fitted preprocessing output differs from the authoritative feature-name metadata.")
        encoder = self.preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        self.categories = {
            feature: [str(value) for value in values.tolist()]
            for feature, values in zip(self.categorical_features, encoder.categories_, strict=True)
        }
        self.model, self.checkpoint_payload = load_frozen_checkpoint(MODEL_PATH)
        self.model.eval()

    def schema(self) -> dict[str, Any]:
        """Describe the form strictly from the fitted Phase 3.1 schema."""
        return {
            "numerical": self.numerical_features,
            "categorical": self.categorical_features,
            "categories": self.categories,
            "transformed_feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "architecture": "58 → 128 → 64 → 32 → 1",
            "parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

    def transform_record(self, values: dict[str, Any]) -> np.ndarray:
        """Apply the already-fitted transformer to one raw-feature record."""
        unexpected = set(values) - set(self.input_features)
        if unexpected:
            raise ValueError("Unexpected fields: " + ", ".join(sorted(unexpected)))
        missing = set(self.input_features) - set(values)
        if missing:
            raise ValueError("Missing fields: " + ", ".join(sorted(missing)))

        row: dict[str, Any] = {}
        for feature in self.numerical_features:
            value = values[feature]
            if value is None or (isinstance(value, str) and value.strip() == ""):
                row[feature] = np.nan
                continue
            if isinstance(value, bool):
                raise ValueError(f"{feature}: enter a numeric value.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{feature}: enter a valid numeric value.") from exc
            if not math.isfinite(number):
                raise ValueError(f"{feature}: value must be finite.")
            row[feature] = number

        for feature in self.categorical_features:
            value = values[feature]
            if value is None or (isinstance(value, str) and value.strip() == ""):
                row[feature] = np.nan
                continue
            category = str(value)
            if category not in self.categories[feature]:
                raise ValueError(f"{feature}: choose one of the categories provided by the fitted training encoder.")
            row[feature] = category

        frame = pd.DataFrame([row], columns=self.input_features)
        matrix = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        if matrix.shape != (1, 58) or not np.isfinite(matrix).all():
            raise ValueError(f"Preprocessing produced invalid model input with shape {matrix.shape}.")
        return matrix

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return a frozen-model probability and the official fixed-threshold class."""
        matrix = self.transform_record(values)
        probability = float(predict_probabilities(self.model, matrix)[0])
        return {
            "probability": probability,
            "predicted_label": int(probability >= self.threshold),
            "threshold": self.threshold,
        }

