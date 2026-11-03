"""Privacy-conscious operational prediction event logging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import ROOT, sha256_file, utc_now

PREDICTION_LOG = ROOT / "MLOPS" / "logs" / "predictions" / "predictions.jsonl"
CHECKPOINT = ROOT / "DEEP_LEARNING" / "models" / "dl_baseline_best.pt"


def current_model_version() -> str | None:
    if not CHECKPOINT.is_file():
        return None
    return f"pcos-dlnn-{sha256_file(CHECKPOINT)[:12]}"


def log_prediction(
    *,
    model_version: str | None,
    status: str,
    inference_duration_ms: float,
    predicted_label: int | None = None,
    probability: float | None = None,
    error_category: str | None = None,
) -> dict[str, Any]:
    """Append only operational outcome fields; raw input and identifiers are never accepted."""
    if status not in {"SUCCESS", "FAILURE"}:
        raise ValueError("Prediction status must be SUCCESS or FAILURE.")
    event: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "model_version": model_version,
        "status": status,
        "inference_duration_ms": max(0.0, float(inference_duration_ms)),
    }
    if status == "SUCCESS":
        if predicted_label not in (0, 1) or probability is None or not 0.0 <= float(probability) <= 1.0:
            raise ValueError("Successful prediction requires binary label and probability in [0, 1].")
        event.update({"predicted_label": int(predicted_label), "probability": float(probability)})
    else:
        event.update({"error_category": error_category or "prediction_error"})
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    return event

