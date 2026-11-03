"""Hash-based model version manifests and guarded local registry operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import REGISTRY_PATH, VERSION_DIR, write_json


def upsert_version_manifest(version: str, metadata: dict[str, Any]) -> Path:
    """Create or update MLOps-owned metadata, refusing artifact identity changes."""
    path = VERSION_DIR / f"{version}.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("artifact_sha256") != metadata.get("artifact_sha256"):
            raise ValueError(f"Model version {version} already exists for a different checkpoint hash.")
        current.setdefault("observed_runs", []).append(metadata["run_observation"])
        # Keep the original model-version identity and training metadata immutable.
        current["quality_gate"] = metadata["quality_gate"]
        current["lifecycle_status"] = metadata["lifecycle_status"]
        current["registry_status"] = metadata["registry_status"]
        current["deployment_status"] = metadata["deployment_status"]
        write_json(path, current)
    else:
        write_json(path, metadata)
    return path


def register_if_eligible(version: str, model_metadata: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    """Register only after PASS; track other candidates without promotion."""
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {"schema_version": 1, "registered_models": [], "production_model": None}
    if registry.get("production_model") is None:
        registry["production_model"] = None
    registry.setdefault("tracked_models", [])
    entries = registry.setdefault("registered_models", [])
    if not gate.get("eligible_for_registration"):
        tracked = next((item for item in registry["tracked_models"] if item.get("model_version") == version), None)
        if tracked is None:
            registry["tracked_models"].append(
                {
                    "model_version": version,
                    "model_artifact_path": model_metadata["artifact_path"],
                    "artifact_sha256": model_metadata["artifact_sha256"],
                    "training_run_id": model_metadata["training_run_id"],
                    "dataset_version": model_metadata["dataset_version"],
                    "evaluation_metrics": model_metadata["test_metrics"],
                    "quality_gate_result": gate["status"],
                    "lifecycle_status": "TRAINED",
                    "lifecycle_history": ["TRAINED"],
                    "registration_timestamp": None,
                    "deployment_status": "NOT_DEPLOYED",
                }
            )
        else:
            tracked["quality_gate_result"] = gate["status"]
        write_json(REGISTRY_PATH, registry)
        return {"status": "NOT_REGISTERED", "reason": f"Quality gate status is {gate.get('status')}; candidate remains TRAINED.", "production_model_changed": False}

    if version not in {entry.get("model_version") for entry in entries}:
        entries.append({
            "model_version": version,
            "model_name": model_metadata["model_name"],
            "training_run_id": model_metadata["training_run_id"],
            "dataset_version": model_metadata["dataset_version"],
            "artifact_path": model_metadata["artifact_path"],
            "artifact_sha256": model_metadata["artifact_sha256"],
            "metrics": model_metadata["test_metrics"],
            "quality_gate_result": gate["status"],
            "registered_at_utc": model_metadata["registered_at_utc"],
            "lifecycle_status": "REGISTERED",
            "lifecycle_history": ["TRAINED", "VALIDATED", "REGISTERED"],
            "deployment_status": "NOT_DEPLOYED",
        })
    tracked = next((item for item in registry["tracked_models"] if item.get("model_version") == version), None)
    if tracked is None:
        registry["tracked_models"].append({
            "model_version": version,
            "model_artifact_path": model_metadata["artifact_path"],
            "artifact_sha256": model_metadata["artifact_sha256"],
            "training_run_id": model_metadata["training_run_id"],
            "dataset_version": model_metadata["dataset_version"],
            "evaluation_metrics": model_metadata["test_metrics"],
            "quality_gate_result": "PASS",
            "lifecycle_status": "REGISTERED",
            "lifecycle_history": ["TRAINED", "VALIDATED", "REGISTERED"],
            "registration_timestamp": model_metadata["registered_at_utc"],
            "deployment_status": "NOT_DEPLOYED",
        })
    else:
        tracked.update({"quality_gate_result": "PASS", "lifecycle_status": "REGISTERED"})
        tracked["lifecycle_history"] = list(dict.fromkeys(tracked.get("lifecycle_history", []) + ["VALIDATED", "REGISTERED"]))
        tracked["registration_timestamp"] = model_metadata["registered_at_utc"]
    write_json(REGISTRY_PATH, registry)
    return {
        "status": "REGISTERED",
        "reason": "Quality gate passed; version registered as a candidate. Deployment readiness is checked separately; no deployment was performed.",
        "production_model_changed": False,
    }


def mark_deployment_ready(version: str) -> None:
    """Advance a registered candidate after independent readiness checks pass."""
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError("Model registry file is missing.")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entry = next((item for item in registry.get("registered_models", []) if item.get("model_version") == version), None)
    tracked = next((item for item in registry.get("tracked_models", []) if item.get("model_version") == version), None)
    if entry is None or tracked is None:
        raise ValueError("A model must be registered before it can become deployment-ready.")
    entry["lifecycle_status"] = "DEPLOYMENT_READY"
    entry["lifecycle_history"] = list(dict.fromkeys(entry.get("lifecycle_history", []) + ["DEPLOYMENT_READY"]))
    tracked["lifecycle_status"] = "DEPLOYMENT_READY"
    tracked["lifecycle_history"] = list(dict.fromkeys(tracked.get("lifecycle_history", []) + ["DEPLOYMENT_READY"]))
    tracked["deployment_status"] = "READY_NOT_DEPLOYED"
    write_json(REGISTRY_PATH, registry)


