"""Post-hoc SHAP explanations for the frozen REAL_ONLY_BASELINE DLNN."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "DEEP_LEARNING" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from evaluate import (  # noqa: E402
    CHECKPOINT_PATH,
    PREDICTIONS_CSV,
    RESULT_JSON,
    X_TEST_PATH,
    Y_TEST_PATH,
    load_frozen_checkpoint,
    predict_probabilities,
)

DL_ROOT = ROOT / "DEEP_LEARNING"
PROCESSED = DL_ROOT / "data" / "processed"
X_TRAIN_PATH = PROCESSED / "X_train.npy"
FEATURE_NAMES_PATH = DL_ROOT / "reports" / "transformed_feature_names.json"
REPORT_DIR = DL_ROOT / "reports"
INDIVIDUAL_DIR = REPORT_DIR / "shap_individual"
IMPORTANCE_CSV = REPORT_DIR / "shap_feature_importance.csv"
LOCAL_CSV = REPORT_DIR / "shap_local_explanations.csv"
SHAP_VALUES_PATH = REPORT_DIR / "shap_values.npy"
SUMMARY_PLOT_PATH = REPORT_DIR / "shap_summary.png"
GLOBAL_PLOT_PATH = REPORT_DIR / "shap_global_importance.png"
METADATA_PATH = REPORT_DIR / "shap_metadata.json"
THRESHOLD = 0.50
SEED = 42
BACKGROUND_SIZE = 100
GRADIENT_NSAMPLES = 200
REPRO_ATOL = 1e-7
REPRO_RTOL = 1e-6
SHAP_ABSOLUTE_TOLERANCE = 0.01  # DeepExplainer's documented additivity-check tolerance.
PREDICTION_CSV_SERIALIZATION_ATOL = 5e-8

INTEGRITY_PATHS = {
    "checkpoint": CHECKPOINT_PATH,
    "X_train": X_TRAIN_PATH,
    "y_train": PROCESSED / "y_train.npy",
    "X_validation": PROCESSED / "X_validation.npy",
    "y_validation": PROCESSED / "y_validation.npy",
    "X_test": X_TEST_PATH,
    "y_test": Y_TEST_PATH,
    "preprocessing_pipeline": DL_ROOT / "models" / "preprocessing_pipeline.joblib",
    "transformed_feature_names": FEATURE_NAMES_PATH,
    "preprocessing_report": REPORT_DIR / "preprocessing_report.json",
    "phase_3_4_predictions": PREDICTIONS_CSV,
    "phase_3_4_evaluation": RESULT_JSON,
}


class ProbabilityModel(nn.Module):
    """Inference wrapper that exposes P(PCOS=1) without editing the checkpoint."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(inputs))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_input_files() -> None:
    missing = [str(path) for path in INTEGRITY_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen Phase 3 artifacts are missing: {missing}")


def _normalize_shap_values(values: Any, n_samples: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected one SHAP output for binary probability, got {len(values)}.")
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.shape != (n_samples, n_features):
        raise ValueError(f"Expected SHAP values {(n_samples, n_features)}, got {array.shape}.")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise FloatingPointError("SHAP values contain NaN or infinite values.")
    return array


def _try_deep_explainer(
    wrapped_model: ProbabilityModel,
    background: torch.Tensor,
    test_tensor: torch.Tensor,
) -> tuple[str, Any | None, np.ndarray | None, str | None]:
    """Probe DeepExplainer additivity; return a diagnostic and any accepted values."""
    try:
        explainer = shap.DeepExplainer(wrapped_model, background)
        probe_count = min(4, len(test_tensor))
        probe_values = explainer.shap_values(test_tensor[:probe_count], check_additivity=True)
        normalized = _normalize_shap_values(probe_values, probe_count, test_tensor.shape[1])
        return "DeepExplainer", explainer, normalized, None
    except Exception as exc:  # Compatibility diagnostic; fallback is documented in metadata.
        detail = f"{type(exc).__name__}: {exc}"
        return "GradientExplainer", None, None, detail


def _compute_values(
    explainer_type: str,
    wrapped_model: ProbabilityModel,
    background: torch.Tensor,
    test_tensor: torch.Tensor,
) -> tuple[np.ndarray, Any | None]:
    if explainer_type == "DeepExplainer":
        explainer = shap.DeepExplainer(wrapped_model, background)
        values = explainer.shap_values(test_tensor, check_additivity=True)
        return _normalize_shap_values(values, len(test_tensor), test_tensor.shape[1]), explainer
    explainer = shap.GradientExplainer(wrapped_model, background)
    values = explainer.shap_values(test_tensor, nsamples=GRADIENT_NSAMPLES)
    return _normalize_shap_values(values, len(test_tensor), test_tensor.shape[1]), explainer


def _expected_probability(wrapped_model: ProbabilityModel, background: torch.Tensor, explainer: Any | None) -> tuple[float, str]:
    if explainer is not None and hasattr(explainer, "expected_value"):
        values = np.asarray(explainer.expected_value).reshape(-1)
        if len(values) == 1 and np.isfinite(values[0]):
            return float(values[0]), "SHAP explainer expected_value"
    with torch.inference_mode():
        expected = wrapped_model(background).detach().cpu().numpy().reshape(-1)
    return float(np.mean(expected)), "mean model probability over training background"


def _select_individual_cases(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, int | None]:
    definitions = {
        "true_negative": (y_true == 0) & (predicted == 0),
        "true_positive": (y_true == 1) & (predicted == 1),
        "false_positive": (y_true == 0) & (predicted == 1),
        "false_negative": (y_true == 1) & (predicted == 0),
    }
    cases: dict[str, int | None] = {}
    for case_type, mask in definitions.items():
        matches = np.flatnonzero(mask)
        cases[case_type] = int(matches[0]) if len(matches) else None
    return cases


def _save_plots(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    importance: pd.DataFrame,
    local_cases: dict[str, int | None],
    baseline_value: float,
) -> dict[str, str | None]:
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=20)
    plt.gcf().set_size_inches(11, 8)
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close("all")

    top = importance.head(15).sort_values("mean_abs_shap", ascending=True, kind="stable")
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top["feature"], top["mean_abs_shap"])
    axis.set_xlabel("Mean absolute SHAP value (probability contribution scale)")
    axis.set_title("Global feature importance — top 15")
    fig.tight_layout()
    fig.savefig(GLOBAL_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    plotted: dict[str, str | None] = {}
    for case_type, test_index in local_cases.items():
        if test_index is None:
            plotted[case_type] = None
            continue
        explanation = shap.Explanation(
            values=shap_values[test_index],
            base_values=baseline_value,
            data=X_test[test_index],
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        figure = plt.gcf()
        path = INDIVIDUAL_DIR / f"case_{case_type}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        plotted[case_type] = str(path.relative_to(ROOT))
    return plotted


def run_explainability() -> dict[str, Any]:
    """Explain all frozen-model test probabilities and save reproducible artifacts."""
    _require_input_files()
    set_seed(SEED)
    print("=" * 61)
    print("PHASE 3.5 - SHAP EXPLAINABILITY")
    print("=" * 61)

    hashes_before = {name: file_sha256(path) for name, path in INTEGRITY_PATHS.items()}
    print("\n[1/6] Loading frozen DL model", flush=True)
    model, checkpoint = load_frozen_checkpoint(CHECKPOINT_PATH)
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(ROOT)}", flush=True)
    print("Architecture: 58 -> 128 -> 64 -> 32 -> 1", flush=True)
    print(f"Parameters: {parameters:,}", flush=True)
    print("Model status: EVALUATION MODE", flush=True)

    print("\n[2/6] Loading explanation data", flush=True)
    X_train = np.load(X_TRAIN_PATH, allow_pickle=False).astype(np.float32, copy=False)
    X_test = np.load(X_TEST_PATH, allow_pickle=False).astype(np.float32, copy=False)
    y_test = np.load(Y_TEST_PATH, allow_pickle=False).astype(np.int64, copy=False)
    feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    phase34_predictions = pd.read_csv(PREDICTIONS_CSV)
    phase34_result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    if X_train.shape != (378, 58) or X_test.shape != (82, 58) or y_test.shape != (82,):
        raise ValueError(f"Unexpected train/test explanation shapes: {X_train.shape}, {X_test.shape}, {y_test.shape}.")
    if len(feature_names) != 58 or len(set(feature_names)) != 58:
        raise ValueError("Authoritative transformed-feature metadata must contain 58 unique names.")
    if len(phase34_predictions) != len(y_test) or not np.array_equal(phase34_predictions["true_label"].to_numpy(), y_test):
        raise ValueError("Phase 3.4 prediction file does not align with the authoritative test labels.")
    if phase34_result.get("threshold") != THRESHOLD:
        raise ValueError("Phase 3.4 threshold differs from the predefined 0.50 case-selection threshold.")

    background_indices = np.sort(np.random.RandomState(SEED).choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False))
    X_background = X_train[background_indices]
    print(f"Training background samples: {len(X_background)} (deterministic sample; seed {SEED})", flush=True)
    print(f"Test samples explained: {len(X_test)} | features: {len(feature_names)}", flush=True)

    print("\n[3/6] SHAP explainer", flush=True)
    wrapped_model = ProbabilityModel(model).eval()
    background_tensor = torch.as_tensor(X_background, dtype=torch.float32, device="cpu")
    test_tensor = torch.as_tensor(X_test, dtype=torch.float32, device="cpu")
    selected_explainer, _, _, deep_compatibility_issue = _try_deep_explainer(wrapped_model, background_tensor, test_tensor)
    if deep_compatibility_issue:
        print(f"DeepExplainer compatibility diagnostic: {deep_compatibility_issue}", flush=True)
        print("Using SHAP GradientExplainer for probability-space explanations.", flush=True)
    print(f"Explainer: {selected_explainer}", flush=True)
    print("Output explained: P(PCOS = 1)", flush=True)
    print(f"SHAP version: {shap.__version__}", flush=True)

    print("\n[4/6] Generating explanations", flush=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    if selected_explainer == "DeepExplainer":
        try:
            set_seed(SEED)
            shap_values, explainer = _compute_values(selected_explainer, wrapped_model, background_tensor, test_tensor)
        except Exception as exc:
            deep_compatibility_issue = f"Full-set {type(exc).__name__}: {exc}"
            selected_explainer = "GradientExplainer"
            print(f"DeepExplainer full-set compatibility diagnostic: {deep_compatibility_issue}", flush=True)
            print("Falling back to SHAP GradientExplainer; output remains P(PCOS = 1).", flush=True)
            set_seed(SEED)
            shap_values, explainer = _compute_values(selected_explainer, wrapped_model, background_tensor, test_tensor)
    else:
        set_seed(SEED)
        shap_values, explainer = _compute_values(selected_explainer, wrapped_model, background_tensor, test_tensor)
    set_seed(SEED)
    repeated_values, _ = _compute_values(selected_explainer, wrapped_model, background_tensor, test_tensor)
    shap_repeat_max_delta = float(np.max(np.abs(shap_values - repeated_values)))
    shap_reproducible = bool(np.allclose(shap_values, repeated_values, atol=REPRO_ATOL, rtol=REPRO_RTOL))
    if not shap_reproducible:
        raise RuntimeError(f"Repeated SHAP values differ beyond tolerance; max delta={shap_repeat_max_delta}.")

    with torch.inference_mode():
        model_probabilities = wrapped_model(test_tensor).cpu().numpy().reshape(-1)
    phase34_probabilities = phase34_predictions["predicted_probability"].to_numpy(dtype=np.float64)
    prediction_delta = float(np.max(np.abs(model_probabilities - phase34_probabilities)))
    predictions_unchanged = bool(np.allclose(model_probabilities, phase34_probabilities, rtol=0.0, atol=PREDICTION_CSV_SERIALIZATION_ATOL))
    if not predictions_unchanged:
        raise RuntimeError(f"Frozen model predictions differ from Phase 3.4 beyond CSV precision tolerance; max delta={prediction_delta}.")
    if not np.array_equal(model_probabilities, predict_probabilities(model, X_test)):
        raise RuntimeError("Probability wrapper output differs from the Phase 3.4 inference utility.")
    predicted_labels = (model_probabilities >= THRESHOLD).astype(np.int64)
    if not np.array_equal(predicted_labels, phase34_predictions["predicted_label"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Predicted labels differ from Phase 3.4 at the unchanged 0.50 threshold.")

    baseline_value, baseline_source = _expected_probability(wrapped_model, background_tensor, explainer)
    residual = model_probabilities - (baseline_value + shap_values.sum(axis=1))
    max_additivity_residual = float(np.max(np.abs(residual)))
    mean_additivity_residual = float(np.mean(np.abs(residual)))
    if selected_explainer == "DeepExplainer":
        additivity = {
            "status": "CHECKED_BY_DEEP_EXPLAINER",
            "max_absolute_residual": max_additivity_residual,
            "mean_absolute_residual": mean_additivity_residual,
            "absolute_tolerance": SHAP_ABSOLUTE_TOLERANCE,
            "baseline_value": baseline_value,
            "baseline_source": baseline_source,
        }
    else:
        additivity = {
            "status": "APPROXIMATE_DIAGNOSTIC_ONLY",
            "max_absolute_residual": max_additivity_residual,
            "mean_absolute_residual": mean_additivity_residual,
            "absolute_tolerance": None,
            "baseline_value": baseline_value,
            "baseline_source": baseline_source,
            "note": "GradientExplainer estimates expectations with sampled gradients and does not expose an exact expected_value; residual is reported descriptively, not treated as a pass/fail criterion.",
        }

    mean_absolute = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_absolute, kind="stable")
    importance = pd.DataFrame(
        {
            "rank": np.arange(1, len(feature_names) + 1),
            "feature": [feature_names[index] for index in order],
            "mean_abs_shap": mean_absolute[order],
        }
    )

    print(f"Samples explained: {len(X_test)} | SHAP values generated: YES", flush=True)
    print("\n[5/6] Global feature importance", flush=True)
    print(f"Top feature: {importance.iloc[0]['feature']} ({importance.iloc[0]['mean_abs_shap']:.8f})", flush=True)
    print("Top 5 features:", flush=True)
    for row in importance.head(5).itertuples(index=False):
        print(f"  {row.rank}. {row.feature}: {row.mean_abs_shap:.8f}", flush=True)

    cases = _select_individual_cases(y_test, predicted_labels)
    local_rows: list[dict[str, Any]] = []
    for case_type, test_index in cases.items():
        if test_index is None:
            continue
        feature_order = np.argsort(-np.abs(shap_values[test_index]), kind="stable")[:10]
        case_rows = []
        for column_index in feature_order:
            item = {
                "case_type": case_type,
                "test_index": test_index,
                "true_label": int(y_test[test_index]),
                "predicted_probability": float(model_probabilities[test_index]),
                "predicted_label": int(predicted_labels[test_index]),
                "feature": feature_names[column_index],
                "feature_value": float(X_test[test_index, column_index]),
                "shap_value": float(shap_values[test_index, column_index]),
                "abs_shap_value": float(abs(shap_values[test_index, column_index])),
            }
            local_rows.append(item)
            case_rows.append(item)
        pd.DataFrame(case_rows).to_csv(INDIVIDUAL_DIR / f"case_{case_type}.csv", index=False)
    local_frame = pd.DataFrame(local_rows, columns=["case_type", "test_index", "true_label", "predicted_probability", "predicted_label", "feature", "feature_value", "shap_value", "abs_shap_value"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_CSV, index=False)
    local_frame.to_csv(LOCAL_CSV, index=False)
    np.save(SHAP_VALUES_PATH, shap_values, allow_pickle=False)
    individual_plots = _save_plots(shap_values, X_test, feature_names, importance, cases, baseline_value)

    hashes_after = {name: file_sha256(path) for name, path in INTEGRITY_PATHS.items()}
    integrity_passed = hashes_before == hashes_after
    if not integrity_passed:
        changed = [name for name in hashes_before if hashes_before[name] != hashes_after[name]]
        raise RuntimeError(f"Frozen input artifacts changed during explanation: {changed}.")

    metadata = {
        "phase": "3.5",
        "experiment": "REAL_ONLY_BASELINE",
        "shap_version": shap.__version__,
        "explainer_type": selected_explainer,
        "deep_explainer_compatibility_issue": deep_compatibility_issue,
        "model_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "model_checkpoint_epoch": int(checkpoint["epoch"]),
        "background_dataset": str(X_TRAIN_PATH.relative_to(ROOT)),
        "background_selection": "np.random.RandomState(42).choice(378, 100, replace=False), sorted by source-array row position",
        "background_size": int(len(background_indices)),
        "background_training_row_positions": background_indices.tolist(),
        "explained_dataset": str(X_TEST_PATH.relative_to(ROOT)),
        "explained_samples": int(len(X_test)),
        "feature_count": int(len(feature_names)),
        "feature_names_source": str(FEATURE_NAMES_PATH.relative_to(ROOT)),
        "output_space": "P(PCOS = 1)",
        "seed": SEED,
        "gradient_explainer_nsamples": GRADIENT_NSAMPLES if selected_explainer == "GradientExplainer" else None,
        "test_prediction_consistency_with_phase_3_4": predictions_unchanged,
        "prediction_phase_3_4_csv_max_absolute_delta": prediction_delta,
        "prediction_csv_serialization_absolute_tolerance": PREDICTION_CSV_SERIALIZATION_ATOL,
        "repeated_shap_values_within_tolerance": shap_reproducible,
        "shap_repeat_max_absolute_delta": shap_repeat_max_delta,
        "reproducibility_absolute_tolerance": REPRO_ATOL,
        "reproducibility_relative_tolerance": REPRO_RTOL,
        "additivity_check": additivity,
        "individual_cases": cases,
        "individual_case_plots": individual_plots,
        "test_data_modified": False,
        "checkpoint_modified": False,
        "training_performed": False,
        "synthetic_data_used": False,
        "ctgan_or_tvae_loaded": False,
        "artifact_integrity_passed": integrity_passed,
        "integrity_hashes_sha256": hashes_after,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n[6/6] Verification", flush=True)
    print(f"Predictions unchanged: {'YES' if predictions_unchanged else 'NO'}", flush=True)
    print(f"Test data unchanged: {'YES' if integrity_passed else 'NO'}", flush=True)
    print(f"Model unchanged: {'YES' if integrity_passed else 'NO'}", flush=True)
    print(f"SHAP reproducibility: {'PASS' if shap_reproducible else 'FAIL'} (max delta {shap_repeat_max_delta:.3g})", flush=True)
    print(f"Additivity diagnostic: {additivity['status']} (max residual {max_additivity_residual:.6f})", flush=True)
    print(f"Local cases: {cases}", flush=True)
    print("=" * 61)
    print("PHASE 3.5 COMPLETE")
    print("=" * 61, flush=True)
    return metadata


if __name__ == "__main__":
    run_explainability()
