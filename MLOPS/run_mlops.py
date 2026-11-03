"""Run the artifact-only MLOps tracking and promotion-readiness workflow."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import run_pipeline  # noqa: E402


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Pipeline status: {result['status']}")
    print(f"Quality gate: {result['quality_gate']['status']}")
    print(f"Registry: {result['registry']['status']}")
    print(f"Deployment readiness: {result['deployment_readiness']['status']}")
    print(f"Deployment: {result['deployment_status']}")
    print(f"Monitoring report: MLOPS/reports/monitoring_latest.json")

