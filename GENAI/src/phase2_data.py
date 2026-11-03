"""Deterministic split and training-fitted data preparation for SDV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "GENAI/data/raw/PCOS_data.csv"
CONFIG = ROOT / "GENAI/configs/phase2.yaml"
OUT = ROOT / "GENAI/data/processed"
NUMERIC_TEXT = ["II    beta-HCG(mIU/mL)", "AMH(ng/mL)"]
CATEGORICAL = ["Blood Group", "Cycle(R/I)", "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"]
TARGET = "PCOS (Y/N)"


def read_partition(name: str) -> pd.DataFrame:
    """Load processed partition while preserving coded categories as strings."""
    cfg = load_config()
    return pd.read_csv(OUT / f"{name}.csv", dtype={c: str for c in CATEGORICAL + [cfg["target_column"]]})


def load_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def prepare() -> dict:
    cfg = load_config()
    raw = pd.read_csv(RAW)
    target = cfg["target_column"]
    excluded = set(cfg["identifier_columns"] + cfg["excluded_columns"])
    # Split positions before any learned transformation. IDs are not feature columns.
    positions = np.arange(len(raw))
    train_idx, holdout_idx = train_test_split(positions, test_size=1-cfg["split"]["train"], stratify=raw[target], random_state=cfg["random_seed"])
    val_share_of_holdout = cfg["split"]["validation"] / (cfg["split"]["validation"] + cfg["split"]["test"])
    val_idx, test_idx = train_test_split(holdout_idx, test_size=1-val_share_of_holdout, stratify=raw.iloc[holdout_idx][target], random_state=cfg["random_seed"])
    indices = {"train": np.sort(train_idx), "validation": np.sort(val_idx), "test": np.sort(test_idx)}
    features = [c for c in raw.columns if c not in excluded]
    if target not in features:
        raise ValueError("Configured target was excluded from the generator schema")
    categorical_features = [c for c in features if c in CATEGORICAL or c == target]
    numeric_features = [c for c in features if c != target and c not in CATEGORICAL and (pd.api.types.is_numeric_dtype(raw[c]) or c in NUMERIC_TEXT)]
    # Remaining object fields are treated as categorical, without assigning meanings.
    categorical_features += [c for c in features if c not in numeric_features and c not in categorical_features]
    train_raw = raw.iloc[indices["train"]][features].copy()
    # Parse numeric strings without cleaning the raw source; trailing dot is normalized as a format repair.
    parse_audit = {}
    for col in NUMERIC_TEXT:
        if col not in train_raw:
            continue
        for part_name, idx in indices.items():
            series = raw.iloc[idx][col].astype("string").str.strip().str.replace(r"\.$", "", regex=True)
            converted = pd.to_numeric(series, errors="coerce")
            parse_audit.setdefault(col, {})[part_name] = {"source_missing": int(raw.iloc[idx][col].isna().sum()), "unparseable_nonmissing": int((raw.iloc[idx][col].notna() & converted.isna()).sum())}
            if part_name == "train":
                train_raw[col] = converted.to_numpy()
    for col in numeric_features:
        if col not in NUMERIC_TEXT:
            train_raw[col] = pd.to_numeric(train_raw[col], errors="coerce")
    # Fit numeric medians and category modes on training rows only.
    numeric_fill = {c: float(train_raw[c].median()) for c in numeric_features}
    categorical_fill = {c: str(train_raw[c].mode(dropna=True).iloc[0]) for c in categorical_features if train_raw[c].notna().any()}
    def transform(part: str, idx: np.ndarray) -> pd.DataFrame:
        frame = raw.iloc[idx][features].copy()
        for col in numeric_features:
            if col in NUMERIC_TEXT:
                values = frame[col].astype("string").str.strip().str.replace(r"\.$", "", regex=True)
                frame[col] = pd.to_numeric(values, errors="coerce")
            else:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan).fillna(numeric_fill[col]).astype(float)
        for col in categorical_features:
            if col in numeric_features:
                continue
            frame[col] = frame[col].map(lambda x: str(x) if pd.notna(x) else np.nan)
            if col in categorical_fill:
                frame[col] = frame[col].fillna(categorical_fill[col])
            # For unseen/empty category, retain explicit missing token (no invented class meaning).
            frame[col] = frame[col].fillna("__MISSING__").astype(str)
        return frame[features]
    splits = {name: transform(name, idx) for name, idx in indices.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, idx in indices.items():
        splits[name].to_csv(OUT / f"{name}.csv", index=False)
    split_rows = pd.DataFrame({"source_row_index_zero_based": np.concatenate(list(indices.values())), "partition": sum(([name]*len(idx) for name,idx in indices.items()), [])})
    split_rows.to_csv(OUT / "split_indices.csv", index=False)
    metadata = {
        "seed": cfg["random_seed"], "stratify_column": target, "requested_ratios": cfg["split"],
        "row_counts": {name: len(idx) for name,idx in indices.items()},
        "class_counts": {name: {str(k): int(v) for k,v in raw.iloc[idx][target].value_counts().sort_index().items()} for name,idx in indices.items()},
        "source_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "source_row_indices_zero_based": {name: indices[name].tolist() for name in indices},
        "generator_features": features, "excluded_identifiers": sorted(set(cfg["identifier_columns"])),
        "excluded_export_candidate": cfg["excluded_columns"],
        "categorical_columns": categorical_features, "numerical_columns": numeric_features,
        "numeric_training_medians": numeric_fill, "categorical_training_modes": categorical_fill,
        "numeric_text_parse_audit": parse_audit,
        "preprocessing_fit_partition": "train only",
        "generator_fit_partition": "train only; validation and test were not passed to synthesizers",
        "target_encoding": "source values preserved as categories 0/1; clinical label semantics not established",
    }
    (OUT / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
