# Deep Learning Prediction Dashboard

## Purpose

This is a local academic inference demo for the frozen `REAL_ONLY_BASELINE` DLNN. It accepts the model's 41 original source features, applies the already-fitted Phase 3.1 preprocessing pipeline, and returns the checkpoint's probability and fixed-threshold classification. It is not part of the official training or evaluation workflow.

## Files added

- `DEEP_LEARNING/dashboard/app.py` — local standard-library HTTP server, bound to `127.0.0.1`.
- `DEEP_LEARNING/dashboard/index.html`, `style.css`, and `app.js` — responsive form and result display.
- `DEEP_LEARNING/dashboard/test_dashboard.py` — inference, validation, route, and consistency checks.
- `DEEP_LEARNING/src/predict.py` — inference-only adapter around existing artifacts.

No existing GENAI or Deep Learning scientific source, result, or model file was edited.

## Existing artifacts consumed

- `DEEP_LEARNING/models/dl_baseline_best.pt` — official frozen Phase 3.3 checkpoint.
- `DEEP_LEARNING/models/preprocessing_pipeline.joblib` — fitted Phase 3.1 `ColumnTransformer`.
- `DEEP_LEARNING/reports/preprocessing_report.json` — source numeric/categorical column schema.
- `DEEP_LEARNING/reports/transformed_feature_names.json` — authoritative 58-feature model ordering.
- `DEEP_LEARNING/reports/test_evaluation.json` — read only for the established threshold (`0.50`).
- `DEEP_LEARNING/reports/training_summary.json` — existing checkpoint verification used by the Phase 3.4 checkpoint loader.

The fitted encoder's categories supply the dropdown choices. The form includes 31 numerical and 10 categorical source fields. It does not request `Sl. No`, `Patient File No.`, or `PCOS (Y/N)`. Category values are shown as recorded codes; the dashboard does not assign code meanings. No medical cutoffs were added.

## Prediction flow

1. The browser sends every displayed source feature; empty values are explicitly represented and are handled by the existing fitted training-only imputers.
2. The inference adapter checks numeric values and restricts categorical entries to categories recorded by the fitted training encoder.
3. It calls `transform` on the existing preprocessor. It never calls `fit` or `fit_transform`.
4. It loads the existing checkpoint using the project's Phase 3.4 frozen-checkpoint loader, which checks the selected epoch and validation loss, and keeps the model in evaluation mode.
5. The model logit is converted to probability with sigmoid. The official fixed threshold of `0.50` maps probability to class; no threshold search occurs.

The server binds only to loopback (`127.0.0.1`) and does not log submitted feature values. The interface includes the academic-prototype disclaimer and makes no clinical validation claim.

## Run locally

From the project root in PowerShell:

```powershell
.venv\Scripts\python.exe DEEP_LEARNING\dashboard\app.py
```

After the server prints its ready message, open:

```text
http://127.0.0.1:8000
```

The first load initializes the saved model and transformer, so wait for the printed URL before opening the page. Stop the server with `Ctrl+C`.

## Verification performed

The dashboard test suite ran six checks successfully:

- frozen checkpoint and fitted transformer load; architecture has 18,305 parameters and 58 transformed inputs;
- a real training-partition row produces a finite probability;
- invalid numeric input, missing payload fields, and unsupported category codes are rejected;
- the local page, schema endpoint, and prediction endpoint respond;
- one existing test record used solely for inference-consistency verification transforms to the existing `X_test` row and reproduces its saved Phase 3.4 probability within `1e-7`.

The only test record was used for that explicitly requested consistency check. No test labels were read by dashboard inference or used for a decision. The ordinary training-row smoke test uses the real training partition.

## Preservation

The dashboard does not train, evaluate the test set, create a split, use synthetic data, or write model/preprocessing/report artifacts. The official checkpoint, preprocessing pipeline, GENAI artifacts, Phase 3.3 training history, Phase 3.4 predictions/evaluation, and Phase 3.5 SHAP reports remain the source of truth and were not modified by the dashboard implementation or checks.
