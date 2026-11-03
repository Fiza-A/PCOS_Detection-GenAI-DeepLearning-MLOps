"""Final evaluation of the frozen REAL_ONLY_BASELINE on the held-out test set."""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "DEEP_LEARNING" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from model import ModelConfig, PCOSDLNN, logits_to_probabilities

DL_ROOT = ROOT / "DEEP_LEARNING"
X_TEST_PATH = DL_ROOT / "data" / "processed" / "X_test.npy"
Y_TEST_PATH = DL_ROOT / "data" / "processed" / "y_test.npy"
CHECKPOINT_PATH = DL_ROOT / "models" / "dl_baseline_best.pt"
REPORT_DIR = DL_ROOT / "reports"
METRICS_CSV = REPORT_DIR / "test_evaluation.csv"
PREDICTIONS_CSV = REPORT_DIR / "test_predictions.csv"
CONFUSION_CSV = REPORT_DIR / "confusion_matrix.csv"
RESULT_JSON = REPORT_DIR / "test_evaluation.json"
THRESHOLD = 0.50
SEED = 42
EXPECTED_X_SHAPE = (82, 58)


def set_evaluation_seed(seed: int = SEED) -> None:
    """Set deterministic seeds for reproducible model reconstruction/inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_frozen_checkpoint(path: Path | str = CHECKPOINT_PATH) -> tuple[PCOSDLNN, dict[str, Any]]:
    """Reconstruct the expected DLNN and strictly load its frozen state."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen Phase 3.3 checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("experiment") != "REAL_ONLY_BASELINE":
        raise ValueError("Checkpoint is not tagged as REAL_ONLY_BASELINE.")
    summary_path = REPORT_DIR / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Phase 3.3 training summary is missing; checkpoint selection cannot be verified.")
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("epoch") != training_summary.get("best_epoch"):
        raise ValueError("Checkpoint epoch does not match the Phase 3.3 best-validation-loss epoch.")
    if not np.isclose(float(payload.get("best_validation_loss", np.nan)), float(training_summary.get("best_validation_loss", np.nan)), rtol=0.0, atol=1e-12):
        raise ValueError("Checkpoint validation loss does not match the Phase 3.3 selected-checkpoint summary.")
    if not np.isfinite(float(payload.get("best_validation_loss", float("nan")))):
        raise ValueError("Checkpoint validation loss is missing or non-finite.")
    config = ModelConfig(**payload["architecture_config"])
    if config != ModelConfig():
        raise ValueError(f"Checkpoint architecture differs from the frozen Phase 3.2 architecture: {config}.")
    model = PCOSDLNN(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    if model.training or any(module.training for module in model.modules() if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d))):
        raise RuntimeError("Model dropout or BatchNorm remained in training mode after model.eval().")
    return model, payload


def load_untouched_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load only the authoritative test arrays for final evaluation."""
    for path in (X_TEST_PATH, Y_TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Authoritative Phase 3.1 test array is missing: {path}")
    X_test = np.load(X_TEST_PATH, allow_pickle=False)
    y_test = np.load(Y_TEST_PATH, allow_pickle=False)
    if X_test.shape != EXPECTED_X_SHAPE or y_test.shape != (EXPECTED_X_SHAPE[0],):
        raise ValueError(f"Expected X_test/y_test shapes {EXPECTED_X_SHAPE}/(82,), got {X_test.shape}/{y_test.shape}.")
    if not np.issubdtype(X_test.dtype, np.number) or not np.isfinite(X_test).all():
        raise ValueError("X_test contains non-numeric or non-finite values.")
    if set(np.unique(y_test).tolist()) != {0, 1}:
        raise ValueError("y_test must contain exactly binary labels 0 and 1.")
    return X_test.astype(np.float32, copy=False), y_test.astype(np.int64, copy=False)


def predict_probabilities(model: PCOSDLNN, X: np.ndarray) -> np.ndarray:
    """Generate probabilities in eval mode with gradient tracking disabled."""
    if model.training:
        raise ValueError("Call model.eval() before test inference.")
    tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device="cpu")
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = logits_to_probabilities(logits)
    result = probabilities.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(result).all() or ((result < 0.0) | (result > 1.0)).any():
        raise FloatingPointError("Predicted probabilities are non-finite or outside [0, 1].")
    return result


def calculate_test_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Calculate fixed-threshold and probability-based metrics without tuning."""
    if threshold != THRESHOLD:
        raise ValueError("Primary test threshold is predefined at exactly 0.50.")
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y_true) != len(probabilities) or set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Truth labels and probabilities must have matching lengths and binary truth labels.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    predicted = (probabilities >= THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel().tolist()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "specificity": specificity,
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("At least one calculated test metric is non-finite.")
    return {"metrics": metrics, "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}, "predicted_labels": predicted}


def _write_reports(
    y_test: np.ndarray,
    probabilities: np.ndarray,
    evaluation: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    deterministic_repeat_check: bool,
) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predicted = evaluation["predicted_labels"]
    counts = {str(label): int(np.sum(y_test == label)) for label in (0, 1)}
    metrics = evaluation["metrics"]
    flat_metrics = [
        (key, value)
        for key, value in metrics.items()
    ] + [
        ("threshold", THRESHOLD),
        ("test_samples", int(len(y_test))),
        ("negative_samples", counts["0"]),
        ("positive_samples", counts["1"]),
    ]
    pd.DataFrame(flat_metrics, columns=["metric", "value"]).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(
        {
            "test_index": np.arange(len(y_test), dtype=np.int64),
            "true_label": y_test,
            "predicted_probability": probabilities,
            "predicted_label": predicted,
        }
    ).to_csv(PREDICTIONS_CSV, index=False)
    pd.DataFrame([evaluation["confusion_matrix"]], columns=["TN", "FP", "FN", "TP"]).to_csv(CONFUSION_CSV, index=False)

    result = {
        "phase": "3.4",
        "experiment": "REAL_ONLY_BASELINE",
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "checkpoint_best_validation_loss": float(checkpoint_payload["best_validation_loss"]),
        "test_sample_count": int(len(y_test)),
        "test_class_counts": {"negative_0": counts["0"], "positive_1": counts["1"]},
        "test_class_percentages": {
            "negative_0": float(counts["0"] / len(y_test) * 100),
            "positive_1": float(counts["1"] / len(y_test) * 100),
        },
        "threshold": THRESHOLD,
        "threshold_policy": "Predefined 0.50; not tuned on test labels.",
        "metrics": metrics,
        "confusion_matrix": evaluation["confusion_matrix"],
        "model_architecture_identifier": "PCOSDLNN_58_128_64_32_1",
        "architecture_config": checkpoint_payload["architecture_config"],
        "seed": SEED,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_evaluation_only": True,
        "test_arrays_modified": False,
        "test_arrays_used_for_threshold_selection": False,
        "training_performed": False,
        "synthetic_data_used": False,
        "ctgan_or_tvae_loaded": False,
        "test_predictions_deterministic_on_repeat": bool(deterministic_repeat_check),
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_evaluation() -> dict[str, Any]:
    """Run the frozen model on the held-out test set once and save actual results."""
    set_evaluation_seed(SEED)
    print("=" * 60)
    print("PHASE 3.4 - FINAL DL TEST EVALUATION")
    print("=" * 60)

    print("\n[1/5] Loading frozen baseline", flush=True)
    model, checkpoint = load_frozen_checkpoint()
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(ROOT)}", flush=True)
    print("Architecture: 58 -> 128 -> 64 -> 32 -> 1", flush=True)
    print(f"Parameters: {parameter_count:,}", flush=True)
    print("Model status: EVALUATION MODE (Dropout disabled; BatchNorm eval statistics)", flush=True)

    print("\n[2/5] Loading untouched test set", flush=True)
    X_test, y_test = load_untouched_test_data()
    negative_count = int(np.sum(y_test == 0))
    positive_count = int(np.sum(y_test == 1))
    print(f"Test samples: {len(y_test)} | features: {X_test.shape[1]}", flush=True)
    print(f"Negative: {negative_count} ({negative_count / len(y_test) * 100:.2f}%)", flush=True)
    print(f"Positive: {positive_count} ({positive_count / len(y_test) * 100:.2f}%)", flush=True)
    print("Test-set modification: NONE", flush=True)

    print("\n[3/5] Generating predictions", flush=True)
    print(f"Threshold: {THRESHOLD:.2f} (predefined; not tuned on test labels)", flush=True)
    probabilities = predict_probabilities(model, X_test)
    repeat_probabilities = predict_probabilities(model, X_test)
    if not np.array_equal(probabilities, repeat_probabilities):
        raise RuntimeError("Repeated inference was not deterministic; evaluation stopped.")
    print(f"Predictions generated: {len(probabilities)} | repeated inference identical: YES", flush=True)

    print("\n[4/5] Final evaluation", flush=True)
    evaluation = calculate_test_metrics(y_test, probabilities, threshold=THRESHOLD)
    for metric, value in evaluation["metrics"].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.6f}", flush=True)

    print("\n[5/5] Confusion matrix", flush=True)
    for key, value in evaluation["confusion_matrix"].items():
        print(f"{key}: {value}", flush=True)
    result = _write_reports(y_test, probabilities, evaluation, checkpoint, deterministic_repeat_check=True)
    print(f"Metrics: {METRICS_CSV.relative_to(ROOT)}", flush=True)
    print(f"Predictions: {PREDICTIONS_CSV.relative_to(ROOT)}", flush=True)
    print(f"Confusion matrix: {CONFUSION_CSV.relative_to(ROOT)}", flush=True)
    print(f"Structured result: {RESULT_JSON.relative_to(ROOT)}", flush=True)
    print("=" * 60)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 60, flush=True)
    return result


if __name__ == "__main__":
    run_evaluation()
