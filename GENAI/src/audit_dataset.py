"""Read-only, reproducible Phase 1 audit for the raw PCOS CSV."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "GENAI" / "data" / "raw" / "PCOS_data.csv"
OUT = ROOT / "GENAI" / "data" / "processed"
TARGET = "PCOS (Y/N)"
IDENTIFIERS = {"Sl. No", "Patient File No."}
CODED_CATEGORICAL = {"Blood Group", "Cycle(R/I)"}
YES_NO_FIELDS = {"Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value.item() if isinstance(value, np.generic) else value


def audit(path: Path = DATA) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read CSV without altering it and compute inventory and quality evidence."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} not found in {path}")
    records: list[dict[str, Any]] = []
    for name in df.columns:
        s = df[name]
        numeric = pd.to_numeric(s, errors="coerce")
        valid_numeric = numeric.notna()
        is_target = name == TARGET
        is_id = name in IDENTIFIERS
        if is_target:
            semantic = "target"
        elif is_id:
            semantic = "identifier"
        elif name in YES_NO_FIELDS or pd.api.types.is_bool_dtype(s.dtype) or (s.nunique(dropna=True) <= 2 and set(s.dropna().unique()).issubset({0, 1, 0.0, 1.0})):
            semantic = "binary"
        elif name in CODED_CATEGORICAL:
            semantic = "categorical"
        elif pd.api.types.is_numeric_dtype(s.dtype):
            semantic = "numerical"
        elif valid_numeric.sum() == s.notna().sum() and s.notna().any():
            semantic = "numerical (stored as text)"
        elif s.nunique(dropna=True) <= 20:
            semantic = "categorical"
        else:
            semantic = "text/unknown"
        if is_id:
            classification = "IDENTIFIER_CANDIDATE"
        elif is_target:
            classification = "TARGET"
        elif semantic.startswith("categorical") or semantic == "binary":
            classification = "CATEGORICAL_CANDIDATE"
        elif semantic.startswith("numerical"):
            classification = "NUMERICAL_CANDIDATE"
        else:
            classification = "UNKNOWN"
        lower_name = name.lower()
        if is_id or name == "Unnamed: 44":
            feature_group = "administrative/unknown"
        elif is_target:
            feature_group = "target"
        elif any(token in lower_name for token in ("hcg", "fsh", "lh", "amh", "tsh", "prl", "vit d", "prg")):
            feature_group = "hormonal/laboratory name-based"
        elif any(token in lower_name for token in ("follicle", "f. size", "endometrium")):
            feature_group = "ultrasound/reproductive name-based"
        elif any(token in lower_name for token in ("cycle", "pregnant", "abortion", "marraige")):
            feature_group = "reproductive name-based"
        elif any(token in lower_name for token in ("gain", "hair", "skin", "pimples", "food", "exercise")):
            feature_group = "symptom/lifestyle name-based"
        elif any(token in lower_name for token in ("age", "weight", "height", "bmi", "waist", "hip", "blood group")):
            feature_group = "demographic/anthropometric name-based"
        elif any(token in lower_name for token in ("pulse", "rr ", "hb(", "bp ", "rbs")):
            feature_group = "measurement name-based"
        else:
            feature_group = "UNKNOWN"
        stats = numeric[valid_numeric]
        records.append({
            "original_name": name,
            "normalized_name_metadata_only": "_".join(name.strip().lower().replace("/", " ").split()),
            "dtype": str(s.dtype), "semantic_type_inferred": semantic,
            "classification": classification, "preliminary_feature_group": feature_group,
            "missing_count": int(s.isna().sum()),
            "missing_percent": round(float(s.isna().mean() * 100), 6),
            "missing_representation": "pandas NA/blank parsed as missing" if s.isna().any() else "no parsed missing values",
            "unique_count_including_missing": int(s.nunique(dropna=False)),
            "sample_values": json.dumps([_json_value(x) for x in s.dropna().drop_duplicates().head(5)], ensure_ascii=False),
            "minimum": _json_value(stats.min()) if not stats.empty else None,
            "maximum": _json_value(stats.max()) if not stats.empty else None,
            "mean": _json_value(stats.mean()) if not stats.empty else None,
            "median": _json_value(stats.median()) if not stats.empty else None,
            "std": _json_value(stats.std()) if not stats.empty else None,
            "q1": _json_value(stats.quantile(.25)) if not stats.empty else None,
            "q3": _json_value(stats.quantile(.75)) if not stats.empty else None,
            "iqr": _json_value(stats.quantile(.75) - stats.quantile(.25)) if not stats.empty else None,
            "zero_count": int((stats == 0).sum()), "negative_count": int((stats < 0).sum()),
            "provenance_confidence": "low; inferred from column name/type only",
        })
    inventory = pd.DataFrame(records)
    duplicate_subset = df.drop(columns=list(IDENTIFIERS & set(df.columns)))
    id_summary = {c: {"unique": int(df[c].nunique(dropna=False)), "duplicate_values": int(df[c].duplicated(keep=False).sum())} for c in IDENTIFIERS if c in df}
    target_counts = df[TARGET].value_counts(dropna=False).to_dict()
    target_copies = [c for c in df.columns if c != TARGET and df[c].equals(df[TARGET])]
    cat_cols = [r["original_name"] for r in records if r["semantic_type_inferred"] in {"binary", "categorical"}]
    categorical_frequencies = {c: {str(_json_value(k)): int(v) for k, v in df[c].value_counts(dropna=False).items()} for c in cat_cols}
    numeric_summary = {}
    for r in records:
        c = r["original_name"]
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if not vals.empty and r["semantic_type_inferred"].startswith("numerical"):
            q1, q3 = vals.quantile([.25, .75])
            low, high = q1 - 1.5 * (q3-q1), q3 + 1.5 * (q3-q1)
            numeric_summary[c] = {"quantiles": {str(q): _json_value(vals.quantile(q)) for q in [.01,.05,.25,.5,.75,.95,.99]}, "iqr_fence_low": float(low), "iqr_fence_high": float(high), "outside_iqr_fence_count": int(((vals < low) | (vals > high)).sum())}
    report = {
        "dataset": {"filename": path.name, "rows": int(df.shape[0]), "columns": int(df.shape[1]), "exact_duplicate_rows": int(df.duplicated().sum()), "duplicates_excluding_identifier_candidates": int(duplicate_subset.duplicated().sum())},
        "target": {"column": TARGET, "dtype": str(df[TARGET].dtype), "counts": {str(_json_value(k)): int(v) for k,v in target_counts.items()}, "percentages": {str(_json_value(k)): round(float(v/len(df)*100), 6) for k,v in target_counts.items()}, "missing": int(df[TARGET].isna().sum()), "invalid_values_outside_0_1": int((~df[TARGET].isin([0,1])).sum())},
        "columns_identical_to_target": target_copies,
        "identifier_candidates": id_summary,
        "categorical_frequencies": categorical_frequencies,
        "numeric_iqr_screen": numeric_summary,
        "column_inventory_rows": len(inventory),
        "audit_scope": "descriptive; no imputation, transformations, row deletion, split, or modeling",
    }
    return inventory, report


def write_outputs(path: Path = DATA) -> None:
    inventory, report = audit(path)
    OUT.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT / "column_inventory.csv", index=False)
    (OUT / "data_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
    print(f"Audit complete: {OUT}")
