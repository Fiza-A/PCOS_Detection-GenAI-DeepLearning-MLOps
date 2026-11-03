"""Refresh the local operational monitoring snapshot from prediction events."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monitoring import aggregate_monitoring, registry_model_state  # noqa: E402


if __name__ == "__main__":
    model_state = registry_model_state()
    report = aggregate_monitoring(**model_state)
    print(f"Total predictions: {report['total_predictions']}")
    print(f"Successful / failed: {report['successful_predictions']} / {report['failed_predictions']}")
    print(f"Average latency (ms): {report['average_inference_latency_ms']}")
    print(f"Positive / negative: {report['prediction_distribution']['positive']} / {report['prediction_distribution']['negative']}")
    print(f"Candidate model version: {report['candidate_model_version']}")
    print(f"Serving model version: {report['serving_model_version']}")
    print(f"Deployment status: {report['deployment_status']}")
    print("Drift readiness:")
    for name, result in report["drift_readiness"].items():
        print(f"  {name}: {result['status']} — {result['reason']}")
    print("MLOPS/reports/monitoring_latest.json")

