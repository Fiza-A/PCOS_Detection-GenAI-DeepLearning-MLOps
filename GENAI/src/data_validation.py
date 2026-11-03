"""Technical/data-structural checks; no clinical rules are asserted."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2_data import ROOT, load_config, read_partition

BASE = ROOT / "GENAI/data/processed"
OUT = ROOT / "GENAI/reports/data_validation"


def validate(generator: str) -> dict:
    g = generator.lower()
    train = read_partition("train")
    cfg = load_config()
    syn = pd.read_csv(ROOT / f"GENAI/data/synthetic/{g}/synthetic_{g}.csv", dtype={c: str for c in ["Blood Group", "Cycle(R/I)", "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", cfg["target_column"]]})
    target = cfg["target_column"]
    cat_cols = ["Blood Group", "Cycle(R/I)", "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", target]
    expected, actual = list(train.columns), list(syn.columns)
    numeric = [c for c in expected if c not in cat_cols]
    unseen, missing_categories, numeric_details = {}, {}, {}
    for c in cat_cols:
        real_set = set(train[c].dropna().astype(str)); syn_set = set(syn[c].dropna().astype(str))
        unseen[c] = sorted(syn_set - real_set); missing_categories[c] = sorted(real_set - syn_set)
    for c in numeric:
        vals = pd.to_numeric(syn[c], errors="coerce")
        ref = pd.to_numeric(train[c], errors="coerce")
        numeric_details[c] = {"dtype": str(syn[c].dtype), "numeric_dtype": bool(pd.api.types.is_numeric_dtype(syn[c])), "non_numeric_count": int(vals.isna().sum()), "non_finite_count": int((~np.isfinite(vals.dropna())).sum()), "below_training_min_count": int((vals < ref.min()).sum()), "above_training_max_count": int((vals > ref.max()).sum()), "min": float(vals.min()), "max": float(vals.max())}
    dtype_compatibility = {c: {"synthetic_dtype": str(syn[c].dtype), "expected_family": "categorical/string" if c in cat_cols else "numeric", "compatible": (syn[c].dtype == object or pd.api.types.is_string_dtype(syn[c])) if c in cat_cols else pd.api.types.is_numeric_dtype(syn[c])} for c in expected if c in syn}
    result = {
        "generator": g.upper(), "expected_columns": expected, "actual_columns": actual,
        "missing_columns": [c for c in expected if c not in actual], "unexpected_columns": [c for c in actual if c not in expected],
        "column_order_matches": expected == actual, "row_count": len(syn),
        "missing_cells_by_column": {c: int(v) for c,v in syn.isna().sum().items()},
        "unseen_categories": unseen, "training_categories_absent_in_synthetic": missing_categories,
        "dtype_compatibility": dtype_compatibility,
        "numerical_support_checks": numeric_details,
        "target_counts_training": {str(k): int(v) for k,v in train[target].value_counts().items()},
        "target_counts_synthetic": {str(k): int(v) for k,v in syn[target].value_counts().items()},
        "target_invalid_values": sorted(set(syn[target].astype(str)) - set(train[target].astype(str))),
        "identifier_columns_present": sorted(set(cfg["identifier_columns"]) & set(actual)),
        "clinical_validity": "Not established from available project sources.",
        "technical_notes": "Out-of-training-range values are reported as support differences, not medically impossible values. No medical ranges are available.",
    }
    hard = (target in actual and len(syn) > 0 and not result["missing_columns"] and not result["unexpected_columns"] and result["column_order_matches"] and not result["identifier_columns_present"] and not result["target_invalid_values"] and not sum(result["missing_cells_by_column"].values()) and all(v["compatible"] for v in dtype_compatibility.values()) and all(not v["non_numeric_count"] and not v["non_finite_count"] for v in numeric_details.values()))
    result["hard_technical_checks_pass"] = bool(hard)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"{g}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
