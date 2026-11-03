"""Local aggregation of privacy-safe prediction events and drift readiness."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from common import ROOT, sha256_file, utc_now, write_json

DEFAULT_PREDICTION_LOG = ROOT / "MLOPS" / "logs" / "predictions" / "predictions.jsonl"
DEFAULT_REPORT = ROOT / "MLOPS" / "reports" / "monitoring_latest.json"
CHECKPOINT = ROOT / "DEEP_LEARNING" / "models" / "dl_baseline_best.pt"
DEFAULT_REGISTRY = ROOT / "MLOPS" / "model_registry" / "registry.json"


def _read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    events: list[dict[str, Any]] = []
    invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    events.append(value)
                else:
                    invalid += 1
            except json.JSONDecodeError:
                invalid += 1
    return events, invalid


def registry_model_state(registry_path: Path = DEFAULT_REGISTRY) -> dict[str, str | None]:
    """Return candidate and genuinely deployed model identities from registry state."""
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "candidate_model_version": None,
            "serving_model_version": None,
            "deployment_status": "NOT_DEPLOYED",
        }

    tracked = registry.get("tracked_models", [])
    registered = registry.get("registered_models", [])
    candidate = None
    if tracked:
        candidate = tracked[-1].get("model_version")
    elif registered:
        candidate = registered[-1].get("model_version")

    production_pointer = registry.get("production_model")
    serving_entry = next(
        (item for item in registered if item.get("model_version") == production_pointer),
        None,
    )
    deployment_status = serving_entry.get("deployment_status", "NOT_DEPLOYED") if serving_entry else "NOT_DEPLOYED"
    serving = production_pointer if production_pointer and deployment_status == "DEPLOYED" else None
    if serving is None and deployment_status == "DEPLOYED":
        deployment_status = "NOT_DEPLOYED"
    return {
        "candidate_model_version": candidate,
        "serving_model_version": serving,
        "deployment_status": deployment_status,
    }


def aggregate_monitoring(
    prediction_log: Path = DEFAULT_PREDICTION_LOG,
    report_path: Path | None = DEFAULT_REPORT,
    candidate_model_version: str | None = None,
    serving_model_version: str | None = None,
    deployment_status: str = "NOT_DEPLOYED",
    reference_available: bool = True,
    ground_truth_available: bool = False,
) -> dict[str, Any]:
    """Summarize requests; never infer model quality or claim drift without evidence."""
    events, invalid_lines = _read_events(prediction_log)
    successes = [event for event in events if event.get("status") == "SUCCESS"]
    failures = [event for event in events if event.get("status") == "FAILURE"]
    latencies = [
        float(event["inference_duration_ms"])
        for event in successes
        if isinstance(event.get("inference_duration_ms"), (int, float))
        and math.isfinite(float(event["inference_duration_ms"]))
        and float(event["inference_duration_ms"]) >= 0
    ]
    predictions = {"negative": 0, "positive": 0}
    for event in successes:
        if event.get("predicted_label") == 0:
            predictions["negative"] += 1
        elif event.get("predicted_label") == 1:
            predictions["positive"] += 1
    total = len(events)
    version_counts: dict[str, int] = {}
    for event in successes:
        version = event.get("model_version") or "unknown"
        version_counts[version] = version_counts.get(version, 0) + 1

    has_production = bool(successes)
    drift = {
        "feature_drift": {
            "status": "INSUFFICIENT_DATA",
            "reason": "Raw clinical features are intentionally not logged; no privacy-safe feature summary/reference comparison is available.",
        },
        "prediction_drift": {
            "status": "INSUFFICIENT_DATA",
            "reason": "No production predictions are available." if not has_production else "Production events exist, but no approved reference comparison threshold/sample policy is configured.",
        },
        "performance_drift": {
            "status": "INSUFFICIENT_DATA",
            "reason": "Ground-truth labels are unavailable for production events." if not ground_truth_available else "No approved performance-drift decision policy is configured.",
        },
    }
    report = {
        "created_at_utc": utc_now(),
        "prediction_log": str(prediction_log.relative_to(ROOT)) if prediction_log.is_relative_to(ROOT) else str(prediction_log),
        "total_predictions": total,
        "successful_predictions": len(successes),
        "failed_predictions": len(failures),
        "success_rate": float(len(successes) / total) if total else None,
        "average_inference_latency_ms": float(sum(latencies) / len(latencies)) if latencies else None,
        "minimum_inference_latency_ms": min(latencies) if latencies else None,
        "maximum_inference_latency_ms": max(latencies) if latencies else None,
        "latest_prediction_timestamp": max((str(event.get("timestamp_utc", "")) for event in events), default=None),
        "candidate_model_version": candidate_model_version,
        "serving_model_version": serving_model_version if deployment_status == "DEPLOYED" else None,
        "deployment_status": deployment_status,
        "prediction_distribution": predictions,
        "prediction_model_version_counts": version_counts,
        "production_performance_metrics": None,
        "production_performance_note": "Production performance metrics unavailable: ground-truth labels are not available." if not ground_truth_available else "Ground truth exists; performance metric computation is not configured in this phase.",
        "drift_readiness": drift,
        "feature_values_logged": False,
        "invalid_log_lines": invalid_lines,
        "reference_data_available": bool(reference_available),
        "ground_truth_available": bool(ground_truth_available),
        "model_artifact_sha256": sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report

