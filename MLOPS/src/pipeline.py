"""Artifact-only MLOps pipeline for the existing real-only DLNN experiment."""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common import (
    CONFIG_PATH,
    DATASET_DIR,
    LOG_PATH,
    MLOPS_ROOT,
    REPORT_DIR,
    ROOT,
    TRACKING_DIR,
    VERSION_DIR,
    composite_hash,
    log_event,
    sha256_file,
    utc_now,
    write_json,
)
from deployment import check_deployment_readiness
from monitoring import aggregate_monitoring, registry_model_state
from quality_gate import evaluate_quality_gate
from registry import mark_deployment_ready, register_if_eligible, upsert_version_manifest


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"MLOps configuration missing: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} artifact is missing: {path.relative_to(ROOT)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} artifact must contain a JSON object: {path.relative_to(ROOT)}")
    return value


def _artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {name: ROOT / relative for name, relative in config["artifacts"].items()}


def _require_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path.relative_to(ROOT)}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required project artifacts are missing: " + "; ".join(missing))


def _read_training_loss(history_path: Path, best_epoch: int) -> float:
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Existing training history is empty.")
    selected = [row for row in rows if int(row["epoch"]) == int(best_epoch)]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one history row for selected epoch {best_epoch}.")
    return float(selected[0]["training_loss"])


def _dataset_metadata(paths: dict[str, Path], preprocessing: dict[str, Any], split_metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
    file_names = (
        "raw_dataset", "split_metadata", "train_partition", "validation_partition", "test_partition",
        "X_train", "y_train", "X_validation", "y_validation", "X_test", "y_test",
    )
    hashes = {name: sha256_file(paths[name]) for name in file_names}
    dataset_digest = composite_hash(hashes)
    prep_info = preprocessing["preprocessing"]
    feature_names_path = paths["transformed_feature_names"]
    transformed_feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    dataset = {
        "dataset_version": f"pcos-real-only-{dataset_digest[:12]}",
        "dataset_sha256": dataset_digest,
        "dataset_name": "PCOS_data.csv / authoritative REAL_ONLY_BASELINE partitions",
        "dataset_location": {
            name: str(paths[name].relative_to(ROOT)) for name in file_names
        },
        "source_hashes_sha256": hashes,
        "raw_rows": int(sum(preprocessing["target"]["class_counts_total"].values())),
        "raw_source_feature_count": len(prep_info["numerical_features"]) + len(prep_info["categorical_features"]),
        "model_feature_count": int(preprocessing["n_features_after_encoding"]),
        "source_feature_names": prep_info["numerical_features"] + prep_info["categorical_features"],
        "transformed_feature_names": transformed_feature_names,
        "target_column": preprocessing["target"]["column"],
        "partition_rows": preprocessing["partition_rows"],
        "preprocessing": {
            "fit_partition": prep_info["fit_partition"],
            "numerical_imputation": prep_info["numerical_imputation"],
            "numerical_scaling": prep_info["numerical_scaling"],
            "categorical_imputation": prep_info["categorical_imputation"],
            "categorical_encoding": prep_info["categorical_encoding"],
        },
        "synthetic_data_used": False,
        "generation_source": "Original real PCOS records; no synthetic rows were used for this DL baseline.",
        "available_synthetic_sources": {
            "CTGAN": {"status": "REVIEW_REQUIRED", "used": False},
            "TVAE": {"status": "REVIEW_REQUIRED", "used": False},
        },
        "split_metadata_version": split_metadata.get("schema_version", "not specified"),
        "recorded_at_utc": utc_now(),
    }
    return dataset, dataset_digest


def _markdown_report(report: dict[str, Any]) -> str:
    metrics = report["test_metrics"]
    gate = report["quality_gate"]
    checks = gate["checks"]
    rows = "\n".join(
        f"| {name} | {checks[name]['observed'] if checks[name]['observed'] is not None else '—'} | "
        f"{checks[name]['minimum'] if checks[name]['minimum'] is not None else 'Not configured'} | "
        f"{checks[name]['status']} |"
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
    )
    reasons = "\n".join(f"- {reason}" for reason in gate["reasons"])
    return f"""# MLOps Pipeline Run

- Pipeline run ID: `{report['run_id']}`
- Training run ID: `{report['training_run_id']}`
- Dataset version: `{report['dataset_version']}`
- Model version: `{report['model_version']}`
- Timestamp (UTC): `{report['timestamp_utc']}`
- Pipeline status: `{report['status']}`
- Training action: `{report['training_action']}`

## Existing-run metrics

- Training: best-epoch training loss `{report['training_metrics']['best_epoch_training_loss']}`.
- Validation: best validation loss `{report['validation_metrics']['best_validation_loss']}`.
- Test metrics are read from the saved Phase 3.4 evaluation artifact; no test evaluation was rerun.

| Metric | Observed | Configured minimum | Gate |
|---|---:|---:|---|
{rows}

## Quality gate: {gate['status']}

{reasons or 'All configured criteria passed.'}

## Registry

- Registration: `{report['registry']['status']}`
- Deployment readiness: `{report['deployment_readiness']['status']}`
- Deployment: `{report['deployment_status']}`
- Production model changed: `{report['registry']['production_model_changed']}`

## Monitoring snapshot

- Prediction count: `{report['monitoring']['total_predictions']}`
- Successful / failed: `{report['monitoring']['successful_predictions']}` / `{report['monitoring']['failed_predictions']}`
- Mean inference latency (ms): `{report['monitoring']['average_inference_latency_ms']}`
- Positive / negative predictions: `{report['monitoring']['prediction_distribution']['positive']}` / `{report['monitoring']['prediction_distribution']['negative']}`
- Drift readiness: feature `{report['drift_readiness']['feature_drift']['status']}`, prediction `{report['drift_readiness']['prediction_drift']['status']}`, performance `{report['drift_readiness']['performance_drift']['status']}`

## Notes

This pipeline ingests existing artifacts only. It does not train, select a checkpoint, change a threshold, modify the official model, or use CTGAN/TVAE data. Quality-gate metrics come from the existing held-out test report for this record; no new test-based model decision or model selection was performed.
"""


def run_pipeline() -> dict[str, Any]:
    """Track and quality-check the existing DL baseline without running training."""
    pipeline_run_id = f"mlops-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    started = utc_now()
    log_event("pipeline_started", run_id=pipeline_run_id, mode="artifact_ingestion_only")
    try:
        config = _load_config()
        paths = _artifact_paths(config)
        _require_files(paths)
        log_event("data_artifacts_verified", run_id=pipeline_run_id, source="existing_project_artifacts")

        preprocessing = _read_json(paths["preprocessing_report"], "preprocessing report")
        if preprocessing.get("status") != "COMPLETE" or preprocessing.get("synthetic_data_used") is not False:
            raise ValueError("The existing preprocessing report does not certify a complete real-only input dataset.")
        if preprocessing.get("matrix_shapes") != {"train": [378, 58], "validation": [81, 58], "test": [82, 58]}:
            raise ValueError("Existing partition matrix shapes differ from the reviewed 378/81/82 baseline.")
        split_metadata = _read_json(paths["split_metadata"], "authoritative split metadata")
        dataset, dataset_digest = _dataset_metadata(paths, preprocessing, split_metadata)
        dataset_path = DATASET_DIR / f"{dataset['dataset_version']}.json"
        if dataset_path.exists():
            existing_dataset = _read_json(dataset_path, "existing dataset version")
            if existing_dataset.get("dataset_sha256") != dataset_digest:
                raise ValueError("Dataset version collision detected; existing metadata hash differs.")
        else:
            write_json(dataset_path, dataset)
        log_event("dataset_version_recorded", run_id=pipeline_run_id, dataset_version=dataset["dataset_version"])

        training_config = _read_json(paths["training_config"], "training configuration")
        training_summary = _read_json(paths["training_summary"], "training summary")
        evaluation = _read_json(paths["test_evaluation"], "test evaluation")
        if training_config.get("experiment") != "REAL_ONLY_BASELINE" or training_summary.get("experiment") != "REAL_ONLY_BASELINE":
            raise ValueError("Existing training artifacts are not tagged as REAL_ONLY_BASELINE.")
        if training_summary.get("synthetic_data_used") is not False or training_config.get("synthetic_data_loaded") is not False:
            raise ValueError("Existing DL training artifacts do not certify real-only training.")
        if evaluation.get("test_data_evaluation_only") is not True:
            raise ValueError("Existing evaluation artifact does not certify test-only evaluation use.")
        if evaluation.get("threshold") != 0.5:
            raise ValueError("Existing Phase 3.4 evaluation threshold is not the established 0.50.")

        checkpoint_hash = sha256_file(paths["checkpoint"])
        preprocessing_hashes = {
            "pipeline": sha256_file(paths["preprocessor"]),
            "report": sha256_file(paths["preprocessing_report"]),
            "feature_names": sha256_file(paths["transformed_feature_names"]),
        }
        preprocessing_version = f"preprocessing-{composite_hash(preprocessing_hashes)[:12]}"
        model_version = f"pcos-dlnn-{checkpoint_hash[:12]}"
        training_run_id = f"real-only-{composite_hash({'checkpoint': checkpoint_hash, 'dataset': dataset_digest})[:16]}"
        best_epoch = int(training_summary["best_epoch"])
        best_train_loss = _read_training_loss(paths["training_history"], best_epoch)
        val_loss = float(training_summary["best_validation_loss"])
        test_metrics = {name: float(value) for name, value in evaluation["metrics"].items()}
        history_summary = {
            "best_epoch": best_epoch,
            "final_epoch": int(training_summary["final_epoch"]),
            "best_epoch_training_loss": best_train_loss,
            "epochs_run": int(training_summary["epochs_run"]),
        }

        training_record_path = TRACKING_DIR / f"{training_run_id}.json"
        training_record = {
            "run_id": training_run_id,
            "timestamp_utc": started,
            "timestamp_note": "MLOps ingestion timestamp; original training timestamp was not recorded in Phase 3.3 artifacts.",
            "status": "SUCCESS",
            "experiment_name": config["experiment_name"],
            "dataset_version": dataset["dataset_version"],
            "model_name": "PCOS DLNN REAL_ONLY_BASELINE",
            "model_version": model_version,
            "model_type": "PCOSDLNN / MLP",
            "training_config": training_config["training_config"],
            "architecture_config": training_config["architecture_config"],
            "loss": training_config["loss"],
            "optimizer": training_config["optimizer"],
            "number_of_training_samples": int(dataset["partition_rows"]["train"]),
            "number_of_validation_samples": int(dataset["partition_rows"]["validation"]),
            "number_of_test_samples": int(dataset["partition_rows"]["test"]),
            "training_metrics": history_summary,
            "validation_metrics": {"best_validation_loss": val_loss, "best_epoch": best_epoch},
            "test_metrics": test_metrics,
            "training_duration_seconds": float(training_summary["training_duration_seconds"]),
            "random_seed": int(training_config["training_config"]["seed"]),
            "preprocessing_version": preprocessing_version,
            "preprocessing_hashes": preprocessing_hashes,
            "feature_information": {
                "source_feature_count": dataset["raw_source_feature_count"],
                "transformed_feature_count": dataset["model_feature_count"],
                "target_column": dataset["target_column"],
                "source_feature_names": dataset["source_feature_names"],
                "transformed_feature_names": dataset["transformed_feature_names"],
            },
            "model_file": str(paths["checkpoint"].relative_to(ROOT)),
            "model_sha256": checkpoint_hash,
            "training_history_file": str(paths["training_history"].relative_to(ROOT)),
            "test_metrics_source": str(paths["test_evaluation"].relative_to(ROOT)),
            "synthetic_data_used": False,
        }
        if training_record_path.exists():
            existing = _read_json(training_record_path, "existing tracked training run")
            if existing.get("model_sha256") != checkpoint_hash or existing.get("dataset_version") != dataset["dataset_version"]:
                raise ValueError("Stable training-run ID collides with different model/dataset artifacts.")
        else:
            write_json(training_record_path, training_record)
        log_event("existing_training_run_ingested", run_id=pipeline_run_id, training_run_id=training_run_id, model_version=model_version)
        log_event("existing_evaluation_loaded", run_id=pipeline_run_id, metrics_source=str(paths["test_evaluation"].relative_to(ROOT)))

        gate = evaluate_quality_gate(test_metrics, config["quality_gate"])
        log_event("quality_gate_completed", run_id=pipeline_run_id, quality_gate=gate["status"], reasons=gate["reasons"])

        now = utc_now()
        version_metadata = {
            "model_version": model_version,
            "model_name": "PCOS DLNN REAL_ONLY_BASELINE",
            "model_type": "PCOSDLNN / MLP",
            "artifact_path": str(paths["checkpoint"].relative_to(ROOT)),
            "artifact_sha256": checkpoint_hash,
            "training_run_id": training_run_id,
            "dataset_version": dataset["dataset_version"],
            "preprocessing_version": preprocessing_version,
            "created_at_utc": now,
            "registered_at_utc": now,
            "created_at_note": "Version record creation time; source training timestamp is not present in the original artifacts.",
            "training_metrics": training_record["training_metrics"],
            "validation_metrics": training_record["validation_metrics"],
            "test_metrics": test_metrics,
            "quality_gate": gate,
            "lifecycle_status": "REGISTERED" if gate["status"] == "PASS" else "TRAINED",
            "registry_status": "ELIGIBLE" if gate["eligible_for_registration"] else "NOT_REGISTERED",
            "deployment_status": "NOT_DEPLOYED",
            "run_observation": {"run_id": pipeline_run_id, "quality_gate_status": gate["status"], "observed_at_utc": now},
        }
        version_path = upsert_version_manifest(model_version, version_metadata)
        registry_result = register_if_eligible(model_version, version_metadata, gate)
        readiness = check_deployment_readiness(
            model_path=paths["checkpoint"],
            preprocessing_path=paths["preprocessor"],
            evaluation_path=paths["test_evaluation"],
            version_metadata_path=version_path,
            expected_model_sha256=checkpoint_hash,
            expected_preprocessing_sha256=preprocessing_hashes["pipeline"],
            quality_gate=gate,
            model_version=model_version,
        )
        if readiness["deployment_ready"]:
            mark_deployment_ready(model_version)
            version_metadata["lifecycle_status"] = "DEPLOYMENT_READY"
            version_metadata["deployment_status"] = "READY_NOT_DEPLOYED"
            version_path = upsert_version_manifest(model_version, version_metadata)
            registry_result["lifecycle_status"] = "DEPLOYMENT_READY"
        else:
            registry_result["lifecycle_status"] = "TRAINED"
        registry_result["deployment_ready"] = readiness["deployment_ready"]
        registry_result["production_model_changed"] = False
        log_event("model_registry_result", run_id=pipeline_run_id, model_version=model_version, registry_status=registry_result["status"])

        model_state = registry_model_state()
        monitoring = aggregate_monitoring(
            candidate_model_version=model_version,
            serving_model_version=model_state["serving_model_version"],
            deployment_status=model_state["deployment_status"] or "NOT_DEPLOYED",
            reference_available=paths["test_predictions"].is_file(),
            ground_truth_available=False,
        )
        log_event("monitoring_snapshot_created", run_id=pipeline_run_id, total_predictions=monitoring["total_predictions"])

        report = {
            "run_id": pipeline_run_id,
            "training_run_id": training_run_id,
            "timestamp_utc": started,
            "completed_at_utc": utc_now(),
            "status": "SUCCESS",
            "pipeline_mode": "existing_artifact_ingestion",
            "training_action": "NOT_RUN; existing Phase 3.3 run ingested, no retraining",
            "dataset_version": dataset["dataset_version"],
            "dataset_metadata_path": str(dataset_path.relative_to(ROOT)),
            "model_version": model_version,
            "model_version_metadata_path": str(version_path.relative_to(ROOT)),
            "model_artifact_path": str(paths["checkpoint"].relative_to(ROOT)),
            "preprocessing_version": preprocessing_version,
            "training_metrics": training_record["training_metrics"],
            "validation_metrics": training_record["validation_metrics"],
            "test_metrics": test_metrics,
            "test_metrics_source": str(paths["test_evaluation"].relative_to(ROOT)),
            "quality_gate": gate,
            "deployment_readiness": readiness,
            "registry": registry_result,
            "deployment_status": "NOT_DEPLOYED",
            "monitoring": monitoring,
            "drift_readiness": monitoring["drift_readiness"],
            "stages": [
                {"name": "DATA", "status": "VERIFIED_FROM_EXISTING_ARTIFACTS"},
                {"name": "PREPROCESSING", "status": "VERIFIED_FROM_EXISTING_ARTIFACTS"},
                {"name": "TRAINING", "status": "EXISTING_RUN_INGESTED; NOT_EXECUTED"},
                {"name": "EVALUATION", "status": "EXISTING_REPORT_INGESTED; NOT_RERUN"},
                {"name": "QUALITY_GATE", "status": gate["status"]},
                {"name": "MODEL_REGISTRY", "status": registry_result["status"]},
                {"name": "DEPLOYMENT_READINESS", "status": readiness["status"]},
                {"name": "DEPLOYMENT", "status": "NOT_DEPLOYED"},
                {"name": "MONITORING", "status": "SNAPSHOT_CREATED"},
                {"name": "DRIFT_READINESS", "status": "INSUFFICIENT_DATA"},
            ],
            "production_model_changed": False,
            "synthetic_data_used": False,
            "official_checkpoint_modified": False,
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        json_path = REPORT_DIR / f"{pipeline_run_id}.json"
        markdown_path = REPORT_DIR / f"{pipeline_run_id}.md"
        write_json(json_path, report)
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")
        log_event("pipeline_completed", run_id=pipeline_run_id, status=report["status"], quality_gate=gate["status"])
        print(f"MLOps run report: {json_path.relative_to(ROOT)}")
        print(f"MLOps summary: {markdown_path.relative_to(ROOT)}")
        return report
    except Exception as exc:
        log_event("pipeline_failed", run_id=pipeline_run_id, status="FAILED", error_type=type(exc).__name__, error=str(exc))
        failed_report = {
            "run_id": pipeline_run_id,
            "timestamp_utc": started,
            "completed_at_utc": utc_now(),
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_action": "NOT_RUN",
            "production_model_changed": False,
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(REPORT_DIR / f"{pipeline_run_id}.json", failed_report)
        raise

