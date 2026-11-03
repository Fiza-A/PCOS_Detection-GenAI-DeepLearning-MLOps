"""Shared paths, serialization, hashing, and structured logging for MLOps."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MLOPS_ROOT = ROOT / "MLOPS"
CONFIG_PATH = MLOPS_ROOT / "configs" / "mlops.yaml"
TRACKING_DIR = MLOPS_ROOT / "experiment_tracking" / "runs"
DATASET_DIR = MLOPS_ROOT / "data_versioning" / "datasets"
VERSION_DIR = MLOPS_ROOT / "model_registry" / "versions"
REGISTRY_PATH = MLOPS_ROOT / "model_registry" / "registry.json"
REPORT_DIR = MLOPS_ROOT / "reports"
LOG_DIR = MLOPS_ROOT / "logs"
LOG_PATH = LOG_DIR / "mlops.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composite_hash(items: dict[str, str]) -> str:
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def log_event(event: str, **fields: Any) -> None:
    """Append one structured event without row-level or patient input data."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": utc_now(), "event": event, **fields}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(record, sort_keys=True, allow_nan=False))

