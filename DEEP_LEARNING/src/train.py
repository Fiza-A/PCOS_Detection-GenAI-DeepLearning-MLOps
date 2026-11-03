"""Controlled real-only DLNN training with validation-based early stopping.

The test arrays are intentionally never opened by this module.
"""
from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "DEEP_LEARNING" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from model import ModelConfig, PCOSDLNN, initialize_model

DL_ROOT = ROOT / "DEEP_LEARNING"
DATA_DIR = DL_ROOT / "data" / "processed"
MODEL_DIR = DL_ROOT / "models"
REPORT_DIR = DL_ROOT / "reports"
PREPROCESSING_REPORT = REPORT_DIR / "preprocessing_report.json"
CHECKPOINT_PATH = MODEL_DIR / "dl_baseline_best.pt"
HISTORY_PATH = REPORT_DIR / "training_history.csv"
CONFIG_PATH = REPORT_DIR / "training_config.json"
SUMMARY_PATH = REPORT_DIR / "training_summary.json"
CURVE_PATH = REPORT_DIR / "training_loss_curve.png"
HISTORY_COLUMNS = ("epoch", "training_loss", "validation_loss", "learning_rate")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15


def configure_determinism(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds and deterministic backend flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_training_validation_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only train and validation arrays; test arrays are never opened."""
    if not PREPROCESSING_REPORT.is_file():
        raise FileNotFoundError("Phase 3.1 preprocessing report is missing; run the approved preprocessing phase first.")
    report = json.loads(PREPROCESSING_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE" or report.get("synthetic_data_used") is not False:
        raise ValueError("Preprocessing report is incomplete or does not certify real-only inputs.")
    expected_shapes = {"train": [378, 58], "validation": [81, 58], "test": [82, 58]}
    if report.get("matrix_shapes") != expected_shapes:
        raise ValueError(f"Preprocessing shapes differ from the approved split: {report.get('matrix_shapes')}.")
    feature_inventory_path = REPORT_DIR / "feature_inventory.csv"
    if not feature_inventory_path.is_file():
        raise FileNotFoundError("Feature inventory is missing; cannot verify target/identifier exclusion.")
    inventory = pd.read_csv(feature_inventory_path)
    excluded = inventory.loc[~inventory["included_in_X"].astype(bool), "name"].tolist()
    if "PCOS (Y/N)" not in excluded or not {"Sl. No", "Patient File No."}.issubset(excluded):
        raise ValueError("Target or identifier exclusion is not confirmed by the saved feature inventory.")

    # Deliberately no X_test.npy / y_test.npy access in Phase 3.3.
    X_train = np.load(DATA_DIR / "X_train.npy", allow_pickle=False).astype(np.float32, copy=False)
    y_train = np.load(DATA_DIR / "y_train.npy", allow_pickle=False).astype(np.int64, copy=False)
    X_validation = np.load(DATA_DIR / "X_validation.npy", allow_pickle=False).astype(np.float32, copy=False)
    y_validation = np.load(DATA_DIR / "y_validation.npy", allow_pickle=False).astype(np.int64, copy=False)
    validate_inputs(X_train, y_train, X_validation, y_validation)
    return X_train, y_train, X_validation, y_validation, report


def validate_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> None:
    """Fail fast on dimensions, invalid values, or non-binary labels."""
    expected = ((378, 58), (81, 58))
    if X_train.shape != expected[0] or X_validation.shape != expected[1]:
        raise ValueError(f"Expected train/validation shapes {expected}, got {X_train.shape}/{X_validation.shape}.")
    if y_train.shape != (378,) or y_validation.shape != (81,):
        raise ValueError(f"Expected label lengths 378/81, got {y_train.shape}/{y_validation.shape}.")
    for name, values in (("X_train", X_train), ("X_validation", X_validation)):
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-numeric, NaN, or infinite values.")
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_validation)) != {0, 1}:
        raise ValueError("Training and validation labels must each contain binary values 0 and 1.")


def calculate_positive_class_weight(y_train: np.ndarray) -> tuple[float, int, int]:
    """Calculate negative/positive ratio from training labels only."""
    labels = np.asarray(y_train)
    if labels.ndim != 1 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_train must be a one-dimensional binary label array.")
    negatives = int(np.sum(labels == 0))
    positives = int(np.sum(labels == 1))
    if negatives == 0 or positives == 0:
        raise ValueError("Both binary classes must occur in y_train to calculate pos_weight.")
    return negatives / positives, negatives, positives


def validate_training_config(config: TrainingConfig) -> None:
    if config.learning_rate != 0.001 or config.batch_size != 32 or config.max_epochs != 100 or config.patience != 15 or config.seed != 42:
        raise ValueError("Phase 3.3 requires the specified seed, optimizer settings, batch size, epochs, and patience.")


def save_checkpoint(
    path: Path | str,
    model: PCOSDLNN,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    best_validation_loss: float,
    class_weight: float,
    negative_count: int,
    positive_count: int,
) -> None:
    """Save sufficient state to reproduce the selected validation checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture_config": asdict(model.config),
            "training_config": asdict(config),
            "seed": config.seed,
            "epoch": int(epoch),
            "best_validation_loss": float(best_validation_loss),
            "positive_class_weight": float(class_weight),
            "training_class_counts": {"negative": int(negative_count), "positive": int(positive_count)},
            "experiment": "REAL_ONLY_BASELINE",
        },
        destination,
    )


def load_checkpoint(path: Path | str, model: PCOSDLNN | None = None) -> tuple[PCOSDLNN, dict[str, Any]]:
    """Load a project-generated checkpoint and restore its model weights."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    restored = model or PCOSDLNN(ModelConfig(**payload["architecture_config"]))
    restored.load_state_dict(payload["model_state_dict"])
    return restored, payload


def validate_history_schema(history: pd.DataFrame) -> None:
    if tuple(history.columns) != HISTORY_COLUMNS:
        raise ValueError(f"Training history columns must be {HISTORY_COLUMNS}.")
    if history.empty:
        raise ValueError("Training history cannot be empty.")
    for column in HISTORY_COLUMNS:
        if not np.isfinite(pd.to_numeric(history[column], errors="raise").to_numpy()).all():
            raise ValueError(f"Training history column {column!r} contains non-finite values.")


def _evaluate_validation(model: PCOSDLNN, X_validation: Tensor, y_validation: Tensor, loss_fn: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X_validation)
        loss = loss_fn(logits, y_validation)
    value = float(loss.item())
    if not np.isfinite(value):
        raise FloatingPointError("Validation loss became NaN or infinite; stopping training.")
    return value


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: TrainingConfig,
    checkpoint_path: Path | str,
    on_epoch: Any | None = None,
) -> dict[str, Any]:
    """Train from train arrays and validate from validation arrays only.

    The signature intentionally has no test-data argument.
    """
    validate_training_config(config)
    validate_inputs(X_train, y_train, X_validation, y_validation)
    destination = Path(checkpoint_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing baseline checkpoint: {destination}")

    configure_determinism(config.seed)
    model = initialize_model(config.seed, ModelConfig(input_dim=X_train.shape[1]))
    device = torch.device("cpu")
    model.to(device)
    class_weight, negatives, positives = calculate_positive_class_weight(y_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32).reshape(-1, 1)),
    )
    loader_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=loader_generator)
    validation_X = torch.from_numpy(np.asarray(X_validation, dtype=np.float32)).to(device)
    validation_y = torch.from_numpy(np.asarray(y_validation, dtype=np.float32).reshape(-1, 1)).to(device)

    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        sum_loss = 0.0
        seen = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Training loss became NaN or infinite at epoch {epoch}; stopping.")
            loss.backward()
            optimizer.step()
            batch_count = len(batch_X)
            sum_loss += float(loss.item()) * batch_count
            seen += batch_count
        training_loss = sum_loss / seen
        if not np.isfinite(training_loss):
            raise FloatingPointError(f"Training loss became NaN or infinite at epoch {epoch}; stopping.")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise FloatingPointError(f"Model parameters became NaN or infinite at epoch {epoch}; stopping.")

        validation_loss = _evaluate_validation(model, validation_X, validation_y, loss_fn)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
        }
        history.append(row)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(destination, model, optimizer, config, epoch, best_validation_loss, class_weight, negatives, positives)
        else:
            epochs_without_improvement += 1

        if on_epoch is not None:
            on_epoch(row, best_epoch, best_validation_loss, epochs_without_improvement)
        if epochs_without_improvement >= config.patience:
            break

    history_frame = pd.DataFrame(history, columns=HISTORY_COLUMNS)
    validate_history_schema(history_frame)
    return {
        "history": history_frame,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_epoch": int(history_frame["epoch"].iloc[-1]),
        "class_weight": class_weight,
        "negative_count": negatives,
        "positive_count": positives,
        "checkpoint_path": str(destination),
    }


def _print_epoch(row: dict[str, float | int], best_epoch: int, best_loss: float, wait: int) -> None:
    print(
        f"Epoch {row['epoch']:03d} | train loss {row['training_loss']:.6f} | "
        f"validation loss {row['validation_loss']:.6f} | lr {row['learning_rate']:.6f} | "
        f"best epoch {best_epoch:03d} ({best_loss:.6f}) | no-improve {wait}",
        flush=True,
    )


def run_training() -> dict[str, Any]:
    """Run the official one-time baseline and save training-only evidence."""
    if CHECKPOINT_PATH.exists():
        raise FileExistsError(f"Baseline checkpoint already exists; refusing a second official run: {CHECKPOINT_PATH}")
    for directory in (MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig()
    print("=" * 62)
    print("PHASE 3.3 - REAL-ONLY DL TRAINING")
    print("=" * 62)
    print("\n[1/6] DATA", flush=True)
    X_train, y_train, X_validation, y_validation, prep_report = load_training_validation_data()
    test_rows = int(prep_report["matrix_shapes"]["test"][0])
    print(f"Train: {X_train.shape[0]} x {X_train.shape[1]}", flush=True)
    print(f"Validation: {X_validation.shape[0]} x {X_validation.shape[1]}", flush=True)
    print(f"Test: {test_rows} (HELD OUT; test arrays not loaded)", flush=True)
    print("Data source: real Phase 3.1 arrays only; synthetic data loaded: NO", flush=True)

    print("\n[2/6] MODEL", flush=True)
    architecture = ModelConfig(input_dim=X_train.shape[1])
    model = initialize_model(config.seed, architecture)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Architecture: DLNN | input features: {architecture.input_dim} | parameters: {parameter_count:,}", flush=True)
    del model

    print("\n[3/6] TRAINING CONFIGURATION", flush=True)
    print(f"Optimizer: Adam | learning rate: {config.learning_rate} | batch size: {config.batch_size}", flush=True)
    print(f"Maximum epochs: {config.max_epochs} | early stopping patience: {config.patience} | seed: {config.seed}", flush=True)
    print("Device: CPU | deterministic algorithms: enabled", flush=True)

    print("\n[4/6] CLASS WEIGHT", flush=True)
    class_weight, negatives, positives = calculate_positive_class_weight(y_train)
    print(f"Negative: {negatives} | positive: {positives}", flush=True)
    print(f"Positive class weight: {negatives} / {positives} = {class_weight:.12f}", flush=True)
    print("Weight calculated using y_train only.", flush=True)

    print("\n[5/6] TRAINING", flush=True)
    start = time.perf_counter()
    result = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        config,
        CHECKPOINT_PATH,
        on_epoch=_print_epoch,
    )
    duration = time.perf_counter() - start
    history: pd.DataFrame = result["history"]
    history.to_csv(HISTORY_PATH, index=False)
    validate_history_schema(history)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["training_loss"], label="Training loss")
    axis.plot(history["epoch"], history["validation_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted BCE loss")
    axis.set_title("REAL_ONLY_BASELINE training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=150)
    plt.close(figure)

    config_payload = {
        "experiment": "REAL_ONLY_BASELINE",
        "training_config": asdict(config),
        "architecture_config": asdict(architecture),
        "loss": "BCEWithLogitsLoss(pos_weight=negative_training_count / positive_training_count)",
        "optimizer": "Adam",
        "class_weight": {"negative_count_train_only": negatives, "positive_count_train_only": positives, "pos_weight": class_weight},
        "input_shapes": {"X_train": list(X_train.shape), "y_train": list(y_train.shape), "X_validation": list(X_validation.shape), "y_validation": list(y_validation.shape), "test_metadata_shape_only": prep_report["matrix_shapes"]["test"]},
        "test_arrays_loaded": False,
        "synthetic_data_loaded": False,
        "deterministic_algorithms": True,
        "device": "cpu",
    }
    CONFIG_PATH.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    summary = {
        "experiment": "REAL_ONLY_BASELINE",
        "best_epoch": result["best_epoch"],
        "best_validation_loss": result["best_validation_loss"],
        "final_epoch": result["final_epoch"],
        "epochs_run": len(history),
        "training_duration_seconds": duration,
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "training_history_path": str(HISTORY_PATH.relative_to(ROOT)),
        "training_curve_path": str(CURVE_PATH.relative_to(ROOT)),
        "test_arrays_loaded": False,
        "test_metrics_calculated": False,
        "synthetic_data_used": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[6/6] CHECKPOINT", flush=True)
    print(f"Best epoch: {result['best_epoch']}", flush=True)
    print(f"Best validation loss: {result['best_validation_loss']:.6f}", flush=True)
    print(f"Final epoch: {result['final_epoch']} | epochs run: {len(history)}", flush=True)
    print(f"Training duration: {duration:.2f} seconds", flush=True)
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(ROOT)}", flush=True)
    print(f"Training history: {HISTORY_PATH.relative_to(ROOT)}", flush=True)
    print(f"Training curve: {CURVE_PATH.relative_to(ROOT)}", flush=True)
    print("\nTEST SET: HELD OUT", flush=True)
    print("No test arrays were loaded and no test metrics were calculated in Phase 3.3.", flush=True)
    print("=" * 62)
    print("PHASE 3.3 COMPLETE")
    print("=" * 62, flush=True)
    return summary


if __name__ == "__main__":
    run_training()
