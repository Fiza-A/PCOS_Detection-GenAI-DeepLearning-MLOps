"""Configurable model quality gate with a fail-closed default policy."""
from __future__ import annotations

import math
from typing import Any

REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
CONFIG_KEYS = {
    "accuracy": "minimum_accuracy",
    "precision": "minimum_precision",
    "recall": "minimum_recall",
    "f1": "minimum_f1",
    "roc_auc": "minimum_roc_auc",
}


def evaluate_quality_gate(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured minima; absent thresholds are insufficient configuration."""
    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    missing_thresholds = False
    metric_failures = False
    for metric_name in REQUIRED_METRICS:
        config_key = CONFIG_KEYS[metric_name]
        threshold = config.get(config_key)
        observed_raw = metrics.get(metric_name)
        try:
            observed = float(observed_raw)
        except (TypeError, ValueError):
            observed = None
        threshold_valid = threshold is not None
        if threshold_valid:
            try:
                threshold = float(threshold)
                threshold_valid = math.isfinite(threshold) and 0.0 <= threshold <= 1.0
            except (TypeError, ValueError):
                threshold_valid = False
        observed_valid = observed is not None and math.isfinite(observed) and 0.0 <= observed <= 1.0

        if not threshold_valid:
            missing_thresholds = True
            check_status = "INSUFFICIENT_CONFIGURATION"
            reason = f"{config_key} is not configured with a valid value in [0, 1]."
        elif not observed_valid:
            metric_failures = True
            check_status = "FAIL"
            reason = f"Observed {metric_name} is missing or invalid."
        else:
            passed = observed >= threshold
            check_status = "PASS" if passed else "FAIL"
            metric_failures = metric_failures or not passed
            reason = "Meets configured minimum." if passed else "Below configured minimum."
        checks[metric_name] = {
            "observed": observed,
            "minimum": threshold if threshold_valid else None,
            "passed": check_status == "PASS",
            "status": check_status,
            "reason": reason,
        }
        if check_status != "PASS":
            reasons.append(f"{metric_name}: {reason}")

    status = "INSUFFICIENT_CONFIGURATION" if missing_thresholds else "FAIL" if metric_failures else "PASS"

    return {
        "status": status,
        "eligible_for_registration": status == "PASS",
        "metric_source": "existing frozen-model evaluation artifact",
        "checks": checks,
        "reasons": reasons,
        "thresholds_configured": not missing_thresholds,
        "threshold_policy": "All five minima must be explicitly configured; no project-source thresholds were available.",
    }

