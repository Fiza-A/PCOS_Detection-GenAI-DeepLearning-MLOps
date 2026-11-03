"""Inference dashboard checks; no training or official-output writes are performed."""
from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DEEP_LEARNING" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import PCOSPredictor  # noqa: E402
from app import DashboardHandler  # noqa: E402


class DashboardInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = PCOSPredictor()

    def test_frozen_checkpoint_and_preprocessor_load(self) -> None:
        self.assertTrue(self.predictor.model.training is False)
        self.assertEqual(self.predictor.schema()["transformed_feature_count"], 58)
        self.assertEqual(self.predictor.schema()["parameters"], 18305)
        self.assertEqual(self.predictor.threshold, 0.50)

    def test_real_training_record_produces_probability(self) -> None:
        train_path = ROOT / "GENAI" / "data" / "processed" / "train.csv"
        train = pd.read_csv(train_path, dtype={name: "string" for name in self.predictor.categorical_features})
        values = train.iloc[0][self.predictor.input_features].to_dict()
        result = self.predictor.predict(values)
        self.assertTrue(np.isfinite(result["probability"]))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIn(result["predicted_label"], (0, 1))

    def test_invalid_numeric_and_missing_fields_are_rejected(self) -> None:
        values = self._training_record()
        numeric = self.predictor.numerical_features[0]
        values[numeric] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "valid numeric"):
            self.predictor.predict(values)
        values = self._training_record()
        values.pop(self.predictor.numerical_features[0])
        with self.assertRaisesRegex(ValueError, "Missing fields"):
            self.predictor.predict(values)

    def test_unseen_category_is_rejected(self) -> None:
        values = self._training_record()
        feature = self.predictor.categorical_features[0]
        values[feature] = "not-an-observed-category"
        with self.assertRaisesRegex(ValueError, "fitted training encoder"):
            self.predictor.predict(values)

    def test_single_test_record_matches_official_transformation_and_prediction(self) -> None:
        """One existing test row is used solely for requested inference consistency verification."""
        processed_test = pd.read_csv(
            ROOT / "GENAI" / "data" / "processed" / "test.csv",
            dtype={name: "string" for name in self.predictor.categorical_features},
        )
        values = processed_test.iloc[0][self.predictor.input_features].to_dict()
        dashboard_matrix = self.predictor.transform_record(values)
        official_matrix = np.load(ROOT / "DEEP_LEARNING" / "data" / "processed" / "X_test.npy", allow_pickle=False)
        np.testing.assert_allclose(dashboard_matrix[0], official_matrix[0], rtol=0.0, atol=1e-6)
        dashboard_probability = self.predictor.predict(values)["probability"]
        official_probability = float(pd.read_csv(ROOT / "DEEP_LEARNING" / "reports" / "test_predictions.csv").iloc[0]["predicted_probability"])
        self.assertAlmostEqual(dashboard_probability, official_probability, places=7)

    def test_local_http_dashboard_serves_schema_and_validation(self) -> None:
        DashboardHandler.predictor = self.predictor
        server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/") as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"PCOS-MLOps", response.read())
            with urlopen(base + "/schema") as response:
                schema = json.loads(response.read())
            self.assertEqual(schema["transformed_feature_count"], 58)
            invalid = self._training_record()
            invalid[self.predictor.numerical_features[0]] = "bad"
            request = Request(base + "/predict", data=json.dumps(invalid).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn(b"valid numeric", caught.exception.read())
            missing = self._training_record()
            missing.pop(self.predictor.categorical_features[0])
            request = Request(base + "/predict", data=json.dumps(missing).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, 400)
            valid = self._training_record()
            request = Request(base + "/predict", data=json.dumps(valid).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request) as response:
                prediction = json.loads(response.read())
            self.assertIn(prediction["predicted_label"], (0, 1))
            self.assertEqual(prediction["threshold"], 0.50)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def _training_record(self) -> dict[str, object]:
        train = pd.read_csv(
            ROOT / "GENAI" / "data" / "processed" / "train.csv",
            dtype={name: "string" for name in self.predictor.categorical_features},
        )
        return train.iloc[0][self.predictor.input_features].to_dict()


if __name__ == "__main__":
    unittest.main()
