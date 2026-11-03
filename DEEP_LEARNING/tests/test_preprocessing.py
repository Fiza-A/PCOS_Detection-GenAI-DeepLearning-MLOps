"""Focused tests for leakage-safe Phase 3.1 data integration."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DEEP_LEARNING" / "src"))
import preprocess_data as prep  # noqa: E402


class PreprocessingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix=".pcos_dl_preprocess_tests_", dir=ROOT / "DEEP_LEARNING")
        cls.output = Path(cls.temp.name) / "run_a"
        cls.report = prep.run_preprocessing(cls.output)
        cls.processed = cls.output / "data" / "processed"
        cls.reports = cls.output / "reports"
        cls.preprocessor = joblib.load(cls.output / "models" / "preprocessing_pipeline.joblib")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_correct_partition_sizes(self) -> None:
        self.assertEqual(self.report["partition_rows"], {"train": 378, "validation": 81, "test": 82})

    def test_02_partition_indices_disjoint_and_exhaustive(self) -> None:
        _, _, metadata, _ = prep.load_authoritative_data()
        sets = [set(metadata["source_row_indices_zero_based"][p]) for p in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(541)))

    def test_03_target_encoding_and_counts(self) -> None:
        raw, partitions, _, _ = prep.load_authoritative_data()
        self.assertEqual(raw[prep.TARGET].value_counts().sort_index().to_dict(), {0: 364, 1: 177})
        self.assertEqual(set(np.unique(np.concatenate([prep._target_vector(partitions[p], p) for p in partitions]))), {0, 1})

    def test_04_identifiers_excluded_from_model_features(self) -> None:
        inventory = pd.read_csv(self.reports / "feature_inventory.csv")
        for name in prep.IDENTIFIERS:
            row = inventory.loc[inventory["name"] == name].iloc[0]
            self.assertFalse(bool(row["included_in_X"]))
            self.assertIn("identifier", row["feature_type"])

    def test_05_target_is_separate_from_X(self) -> None:
        inventory = pd.read_csv(self.reports / "feature_inventory.csv")
        row = inventory.loc[inventory["name"] == prep.TARGET].iloc[0]
        self.assertFalse(bool(row["included_in_X"]))
        self.assertIn("target", row["feature_type"])
        for part in ("train", "validation", "test"):
            y = np.load(self.processed / f"y_{part}.npy", allow_pickle=False)
            self.assertEqual(set(np.unique(y)), {0, 1})

    def test_06_no_nan_values_remain(self) -> None:
        for part in ("train", "validation", "test"):
            self.assertFalse(np.isnan(np.load(self.processed / f"X_{part}.npy", allow_pickle=False)).any())
        self.assertEqual(self.report["saved_partition_missing_counts_for_model_features"], {"train": 0, "validation": 0, "test": 0})

    def test_07_no_infinite_values_remain(self) -> None:
        for part in ("train", "validation", "test"):
            self.assertTrue(np.isfinite(np.load(self.processed / f"X_{part}.npy", allow_pickle=False)).all())

    def test_08_equal_feature_dimensions(self) -> None:
        matrices = [np.load(self.processed / f"X_{part}.npy", allow_pickle=False) for part in ("train", "validation", "test")]
        self.assertEqual(len({matrix.shape[1] for matrix in matrices}), 1)
        self.assertEqual(matrices[0].shape[1], self.report["n_features_after_encoding"])

    def test_09_consistent_saved_feature_order(self) -> None:
        names = json.loads((self.reports / "transformed_feature_names.json").read_text(encoding="utf-8"))
        self.assertEqual(len(names), self.report["n_features_after_encoding"])
        self.assertEqual(len(names), len(set(names)))
        for part in ("train", "validation", "test"):
            matrix = np.load(self.processed / f"X_{part}.npy", allow_pickle=False)
            self.assertEqual(matrix.shape[1], len(names))

    def test_10_preprocessing_parameters_come_from_training_only(self) -> None:
        _, partitions, metadata, _ = prep.load_authoritative_data()
        numeric = metadata["numerical_columns"]
        categorical = [column for column in metadata["categorical_columns"] if column != prep.TARGET]
        numeric_imputer = self.preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        expected_medians = partitions["train"][numeric].median(axis=0).to_numpy(dtype=float)
        np.testing.assert_allclose(numeric_imputer.statistics_, expected_medians)
        encoder = self.preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        for column, observed in zip(categorical, encoder.categories_):
            expected = np.sort(partitions["train"][column].dropna().astype(str).unique())
            np.testing.assert_array_equal(observed.astype(str), expected)
        # A validation-only unseen category is ignored and cannot alter fitted categories/schema.
        heldout = partitions["validation"].loc[:, numeric + categorical].copy()
        heldout.loc[heldout.index[0], categorical[0]] = "__VALIDATION_ONLY_CATEGORY__"
        transformed = self.preprocessor.transform(heldout)
        self.assertEqual(transformed.shape[1], self.report["n_features_after_encoding"])
        self.assertNotIn("__VALIDATION_ONLY_CATEGORY__", set(encoder.categories_[0].astype(str)))

    def test_11_preprocessing_is_reproducible(self) -> None:
        second_output = Path(self.temp.name) / "run_b"
        second_report = prep.run_preprocessing(second_output)
        self.assertEqual(self.report["n_features_after_encoding"], second_report["n_features_after_encoding"])
        for filename in ("X_train.npy", "X_validation.npy", "X_test.npy", "y_train.npy", "y_validation.npy", "y_test.npy"):
            np.testing.assert_array_equal(
                np.load(self.processed / filename, allow_pickle=False),
                np.load(second_output / "data" / "processed" / filename, allow_pickle=False),
            )
        self.assertEqual(
            json.loads((self.reports / "transformed_feature_names.json").read_text(encoding="utf-8")),
            json.loads((second_output / "reports" / "transformed_feature_names.json").read_text(encoding="utf-8")),
        )

    def test_12_raw_source_integrity(self) -> None:
        self.assertEqual(prep._sha256(prep.RAW_PATH), self.report["source_artifacts"]["raw_sha256"])
        self.assertEqual(self.report["raw_dataset_sha256_after_processing"], self.report["source_artifacts"]["raw_sha256"])

    def test_leakage_checks_all_pass_and_no_synthetic_data_used(self) -> None:
        self.assertTrue(all(self.report["leakage_checks"].values()))
        self.assertFalse(self.report["synthetic_data_used"])


if __name__ == "__main__":
    unittest.main()
