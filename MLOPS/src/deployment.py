"""Read-only deployment-readiness checks for a registered model candidate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import REGISTRY_PATH, ROOT, sha256_file


def check_deployment_readiness(
    model_path: Path,
    preprocessing_path: Path,
    evaluation_path: Path,
    version_metadata_path: Path,
    registry_path: Path = REGISTRY_PATH,
    expected_model_sha256: str | None = None,
    expected_preprocessing_sha256: str | None = None,
    quality_gate: dict[str, Any] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Require artifacts, matching hashes, evaluation, gate PASS, and registry record."""
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, reason: str) -> None:
        checks[name] = {"passed": bool(passed), "reason": reason}

    model_hash = None
    prep_hash = None
    if model_path.is_file():
        model_hash = sha256_file(model_path)
        add("model_artifact_exists", True, "Frozen model artifact exists.")
    else:
        add("model_artifact_exists", False, f"Missing model artifact: {model_path}.")
    if preprocessing_path.is_file():
        prep_hash = sha256_file(preprocessing_path)
        add("preprocessing_artifact_exists", True, "Fitted preprocessing artifact exists.")
    else:
        add("preprocessing_artifact_exists", False, f"Missing preprocessing artifact: {preprocessing_path}.")
    add(
        "model_hash_valid",
        model_hash is not None and expected_model_sha256 is not None and model_hash == expected_model_sha256,
        "Model SHA-256 matches recorded metadata." if model_hash is not None and model_hash == expected_model_sha256 else "Model SHA-256 is missing or differs from recorded metadata.",
    )
    add(
        "preprocessing_hash_valid",
        prep_hash is not None and expected_preprocessing_sha256 is not None and prep_hash == expected_preprocessing_sha256,
        "Preprocessing SHA-256 matches recorded metadata." if prep_hash is not None and prep_hash == expected_preprocessing_sha256 else "Preprocessing SHA-256 is missing or differs from recorded metadata.",
    )

    evaluation = None
    if evaluation_path.is_file():
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            evaluation = None
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    valid_metrics = isinstance(metrics, dict) and all(
        isinstance(metrics.get(name), (int, float)) and 0 <= float(metrics[name]) <= 1 for name in metric_names
    )
    add("evaluation_metrics_exist", bool(valid_metrics), "Required evaluation metrics are present." if valid_metrics else "Evaluation artifact or required metrics are missing/invalid.")

    gate_pass = isinstance(quality_gate, dict) and quality_gate.get("status") == "PASS"
    add("quality_gate_passes", gate_pass, "Configured quality gate passed." if gate_pass else f"Quality gate status is {quality_gate.get('status') if isinstance(quality_gate, dict) else 'missing'}; PASS is required.")

    registry_valid = False
    try:
        metadata = json.loads(version_metadata_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        metadata_valid = all(metadata.get(key) for key in ("model_version", "artifact_path", "artifact_sha256", "training_run_id", "dataset_version"))
        registered = any(item.get("model_version") == model_version and item.get("artifact_sha256") == model_hash for item in registry.get("registered_models", []))
        registry_valid = bool(metadata_valid and registered and metadata.get("model_version") == model_version)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        registry_valid = False
    add("registry_metadata_valid", registry_valid, "Version manifest and registry entry match this artifact." if registry_valid else "Valid registered-model metadata for this model version is missing or inconsistent.")

    reasons = [f"{name}: {check['reason']}" for name, check in checks.items() if not check["passed"]]
    return {
        "deployment_ready": not reasons,
        "status": "READY" if not reasons else "NOT_READY",
        "model_version": model_version,
        "checks": checks,
        "reasons": reasons,
        "deployment_performed": False,
    }

