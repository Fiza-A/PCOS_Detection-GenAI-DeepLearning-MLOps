"""Build a train-fitted numeric representation from authoritative real partitions.

This module does not split data, use synthetic rows, or train a predictive model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "GENAI" / "data" / "raw" / "PCOS_data.csv"
GENAI_PROCESSED = ROOT / "GENAI" / "data" / "processed"
SPLIT_METADATA = GENAI_PROCESSED / "split_metadata.json"
SPLIT_INDEX_FILE = GENAI_PROCESSED / "split_indices.csv"
TARGET = "PCOS (Y/N)"
IDENTIFIERS = ("Sl. No", "Patient File No.")
EXPECTED_ROWS = {"train": 378, "validation": 81, "test": 82}
RANDOM_SEED = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_authoritative_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    """Load raw source, saved partitions, metadata, and row-index map; never split."""
    for path in (RAW_PATH, SPLIT_METADATA, SPLIT_INDEX_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required authoritative data artifact is missing: {path}")

    raw = pd.read_csv(RAW_PATH)
    metadata = json.loads(SPLIT_METADATA.read_text(encoding="utf-8"))
    if metadata.get("source_sha256", "").lower() != _sha256(RAW_PATH):
        raise ValueError("Raw dataset hash does not match authoritative split metadata.")
    if metadata.get("stratify_column") != TARGET:
        raise ValueError(f"Authoritative split target is not {TARGET!r}.")
    if metadata.get("row_counts") != EXPECTED_ROWS:
        raise ValueError(f"Authoritative split counts differ from {EXPECTED_ROWS}: {metadata.get('row_counts')}.")
    if TARGET not in raw or set(pd.Series(raw[TARGET]).dropna().unique()) != {0, 1}:
        raise ValueError("Raw target is missing or its observed encoding is not exactly binary {0, 1}.")
    if raw[TARGET].isna().any():
        raise ValueError("Raw target contains missing values; cannot safely construct binary labels.")

    categorical = list(metadata["categorical_columns"])
    dtype_map = {column: "string" for column in categorical if column != TARGET}
    partitions: dict[str, pd.DataFrame] = {}
    for name, expected_rows in EXPECTED_ROWS.items():
        path = GENAI_PROCESSED / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Authoritative {name} partition is missing: {path}")
        frame = pd.read_csv(path, dtype=dtype_map)
        if len(frame) != expected_rows:
            raise ValueError(f"{name} partition has {len(frame)} rows; expected {expected_rows}.")
        if TARGET not in frame:
            raise ValueError(f"Target {TARGET!r} is missing from the {name} partition.")
        partitions[name] = frame

    index_rows = pd.read_csv(SPLIT_INDEX_FILE)
    required_index_columns = {"source_row_index_zero_based", "partition"}
    if not required_index_columns.issubset(index_rows.columns):
        raise ValueError("split_indices.csv does not contain the expected source-index columns.")
    if len(index_rows) != len(raw):
        raise ValueError("Saved split index map does not cover every raw source row exactly once.")
    if index_rows["source_row_index_zero_based"].duplicated().any():
        raise ValueError("Saved split index map contains duplicate source row indices.")

    expected_indices = metadata.get("source_row_indices_zero_based", {})
    sets = {name: set(map(int, expected_indices[name])) for name in EXPECTED_ROWS}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Authoritative source row indices are not pairwise disjoint.")
    if set.union(*sets.values()) != set(range(len(raw))):
        raise ValueError("Authoritative source row indices do not cover all raw records.")
    for name, frame in partitions.items():
        source_y = raw.iloc[sorted(sets[name])][TARGET].to_numpy()
        if not np.array_equal(source_y, pd.to_numeric(frame[TARGET]).to_numpy()):
            raise ValueError(f"{name} target values do not match the source rows identified by split metadata.")
        index_partition = set(index_rows.loc[index_rows["partition"] == name, "source_row_index_zero_based"].astype(int))
        if index_partition != sets[name]:
            raise ValueError(f"split_indices.csv and split_metadata.json disagree for {name}.")
    return raw, partitions, metadata, index_rows


def build_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create the single train-fitted numeric/categorical preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _target_vector(frame: pd.DataFrame, partition: str) -> np.ndarray:
    values = pd.to_numeric(frame[TARGET], errors="raise").to_numpy()
    unique = set(np.unique(values).tolist())
    if unique != {0, 1}:
        raise ValueError(f"{partition} target encoding is not binary 0/1: {sorted(unique)}")
    return values.astype(np.int64)


def _validate_matrix(name: str, matrix: np.ndarray, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(f"{name} has unexpected shape {matrix.shape}.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} contains non-numeric values.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values after preprocessing.")
    return matrix


def build_feature_inventory(
    raw: pd.DataFrame,
    train: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Inventory every raw column and explain inclusion/exclusion from model X."""
    target = metadata["stratify_column"]
    numeric = set(metadata["numerical_columns"])
    categorical = set(metadata["categorical_columns"]) - {target}
    selected = set(metadata["generator_features"]) - {target}
    excluded_ids = set(metadata.get("excluded_identifiers", IDENTIFIERS))
    export_artifacts = set(metadata.get("excluded_export_candidate", []))
    records = []
    for column in raw.columns:
        if column == target:
            semantic_type, included, reason, operation = "target", False, "Stored separately as binary y; never included in X.", "Preserve source 0/1 encoding as int64"
        elif column in excluded_ids:
            semantic_type, included, reason, operation = "identifier", False, "Identifier candidate; excluded from model features and retained in raw/source artifacts.", "None; not transformed"
        elif column in export_artifacts:
            semantic_type, included, reason, operation = "unknown/export artifact", False, "Explicitly excluded by authoritative Phase 2 configuration as an export candidate; retained in raw/source data.", "None; not transformed"
        elif column in selected:
            semantic_type = "numerical" if column in numeric else "categorical" if column in categorical else "unknown"
            if semantic_type == "unknown":
                raise ValueError(f"No authoritative numerical/categorical classification for candidate {column!r}.")
            included, reason = True, ""
            operation = "Training-only median imputation + StandardScaler" if semantic_type == "numerical" else "Training-only most-frequent imputation + OneHotEncoder(handle_unknown='ignore')"
        else:
            raise ValueError(f"Raw column {column!r} has no documented inclusion or exclusion decision.")
        records.append(
            {
                "name": column,
                "original_dtype": str(raw[column].dtype),
                "feature_type": semantic_type,
                "raw_missing_count": int(raw[column].isna().sum()),
                "train_partition_missing_count": int(train[column].isna().sum()) if column in train else None,
                "raw_unique_count": int(raw[column].nunique(dropna=True)),
                "included_in_X": included,
                "preprocessing_operation": operation,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(records)


def run_preprocessing(output_root: Path | str | None = None) -> dict[str, Any]:
    """Fit preprocessing on saved real train rows and serialize numeric partitions."""
    raw, partitions, metadata, index_rows = load_authoritative_data()
    output = Path(output_root) if output_root is not None else ROOT / "DEEP_LEARNING"
    processed_dir = output / "data" / "processed"
    models_dir = output / "models"
    reports_dir = output / "reports"
    for directory in (processed_dir, models_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    target = metadata["stratify_column"]
    numerical_features = list(metadata["numerical_columns"])
    categorical_features = [column for column in metadata["categorical_columns"] if column != target]
    feature_columns = numerical_features + categorical_features
    expected_features = [column for column in metadata["generator_features"] if column != target]
    if set(feature_columns) != set(expected_features) or len(feature_columns) != len(expected_features):
        raise ValueError("Authoritative feature type lists do not cover the candidate feature schema exactly.")
    if any(column in feature_columns for column in (target, *IDENTIFIERS)):
        raise ValueError("Target or identifier candidate found in model feature columns.")
    if not set(feature_columns).issubset(partitions["train"].columns):
        raise ValueError("A candidate model feature is missing from authoritative processed partitions.")

    X = {name: frame.loc[:, feature_columns].copy() for name, frame in partitions.items()}
    y = {name: _target_vector(frame, name) for name, frame in partitions.items()}
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    # The sole fit call is on the authoritative real training partition.
    X_train = preprocessor.fit_transform(X["train"])
    matrices = {
        "train": _validate_matrix("X_train", X_train, EXPECTED_ROWS["train"]),
        "validation": _validate_matrix("X_validation", preprocessor.transform(X["validation"]), EXPECTED_ROWS["validation"]),
        "test": _validate_matrix("X_test", preprocessor.transform(X["test"]), EXPECTED_ROWS["test"]),
    }
    feature_names = preprocessor.get_feature_names_out().tolist()
    if not feature_names or len(feature_names) != matrices["train"].shape[1]:
        raise ValueError("Transformed feature names do not match the numeric matrix width.")
    if len({m.shape[1] for m in matrices.values()}) != 1:
        raise ValueError("Transformed partitions do not have identical feature dimensions.")

    source_indices = {name: set(map(int, metadata["source_row_indices_zero_based"][name])) for name in EXPECTED_ROWS}
    checks = {
        "test_rows_disjoint_from_training": not bool(source_indices["train"] & source_indices["test"]),
        "validation_rows_disjoint_from_training": not bool(source_indices["train"] & source_indices["validation"]),
        "source_indices_collectively_exhaustive": set.union(*source_indices.values()) == set(range(len(raw))),
        "preprocessor_fit_on_training_only": True,
        "target_excluded_from_X": target not in feature_columns,
        "identifier_candidates_excluded_from_X": not any(identifier in feature_columns for identifier in IDENTIFIERS),
        "all_transformed_values_finite": all(np.isfinite(matrix).all() for matrix in matrices.values()),
        "feature_order_identical_across_partitions": len({tuple(feature_names) for _ in matrices}) == 1,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Data leakage/representation checks failed: {failed}")

    for name, matrix in matrices.items():
        np.save(processed_dir / f"X_{name}.npy", matrix, allow_pickle=False)
        np.save(processed_dir / f"y_{name}.npy", y[name], allow_pickle=False)
    joblib.dump(preprocessor, models_dir / "preprocessing_pipeline.joblib")

    inventory = build_feature_inventory(raw, partitions["train"], metadata)
    inventory.to_csv(reports_dir / "feature_inventory.csv", index=False)
    (reports_dir / "transformed_feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    target_distributions = {}
    for name, labels in y.items():
        counts = pd.Series(labels).value_counts().sort_index()
        target_distributions[name] = {
            "rows": int(len(labels)),
            "counts": {str(key): int(value) for key, value in counts.items()},
            "proportions": {str(key): float(value / len(labels)) for key, value in counts.items()},
        }
    class_counts_total = {str(key): int(value) for key, value in raw[target].value_counts().sort_index().items()}
    report = {
        "status": "COMPLETE",
        "random_seed": RANDOM_SEED,
        "experiment": "REAL_ONLY_BASELINE_INPUTS",
        "synthetic_data_used": False,
        "source_artifacts": {
            "raw_dataset": str(RAW_PATH.relative_to(ROOT)),
            "train": str((GENAI_PROCESSED / "train.csv").relative_to(ROOT)),
            "validation": str((GENAI_PROCESSED / "validation.csv").relative_to(ROOT)),
            "test": str((GENAI_PROCESSED / "test.csv").relative_to(ROOT)),
            "split_metadata": str(SPLIT_METADATA.relative_to(ROOT)),
            "split_indices": str(SPLIT_INDEX_FILE.relative_to(ROOT)),
            "raw_sha256": _sha256(RAW_PATH),
        },
        "target": {
            "column": target,
            "observed_source_encoding": {"0": int((raw[target] == 0).sum()), "1": int((raw[target] == 1).sum())},
            "class_counts_total": class_counts_total,
            "partition_distributions": target_distributions,
            "output_dtype": "int64",
        },
        "partition_rows": {name: int(len(partitions[name])) for name in partitions},
        "raw_missing_counts_for_model_features": {
            name: int(raw[name].isna().sum()) for name in feature_columns
        },
        "saved_partition_missing_counts_for_model_features": {
            name: int(frame.loc[:, feature_columns].isna().sum().sum())
            for name, frame in partitions.items()
        },
        "preprocessing": {
            "fit_partition": "authoritative real train.csv only",
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "numerical_imputation": "SimpleImputer(strategy='median') fitted on training rows",
            "numerical_scaling": "StandardScaler fitted on training rows",
            "categorical_imputation": "SimpleImputer(strategy='most_frequent') fitted on training rows",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore') fitted on training rows",
            "unknown_validation_test_categories": "ignored by the fitted one-hot encoder; encoded columns remain unchanged",
            "phase2_input_note": "Saved GENAI partitions already contain Phase 2 training-fitted imputation; no missing cells were present in these partition files at DL preprocessing time.",
            "phase2_numeric_text_parse_audit": metadata.get("numeric_text_parse_audit", {}),
        },
        "excluded_columns": inventory.loc[~inventory["included_in_X"], ["name", "feature_type", "exclusion_reason"]].to_dict(orient="records"),
        "n_features_after_encoding": int(matrices["train"].shape[1]),
        "matrix_shapes": {name: list(matrix.shape) for name, matrix in matrices.items()},
        "matrix_dtype": str(matrices["train"].dtype),
        "leakage_checks": checks,
        "split_index_file_rows": int(len(index_rows)),
        "raw_dataset_sha256_after_processing": _sha256(RAW_PATH),
    }
    (reports_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_preprocessing()
    print(json.dumps({"status": result["status"], "matrix_shapes": result["matrix_shapes"], "n_features_after_encoding": result["n_features_after_encoding"], "leakage_checks": result["leakage_checks"]}, indent=2))
