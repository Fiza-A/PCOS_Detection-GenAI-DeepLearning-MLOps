# MLOPS — Local Tracking, Quality Gate, and Model Registry

## Purpose and current boundary

MLOPS consumes outputs from the existing GENAI and DEEP_LEARNING components. It does not duplicate their algorithms. The current MLOPS workflow adds local dataset lineage, experiment-run records, content-hash model versions, a configurable quality gate, a guarded registry catalog, structured logs, and per-run summaries.

The first execution is **artifact ingestion only**. It verifies and records the already-completed `REAL_ONLY_BASELINE`; it does not call `GENAI/run_genai.py`, `DEEP_LEARNING/run_dl.py`, `train.py`, or `evaluate.py`. CTGAN and TVAE remain `REVIEW_REQUIRED` and are recorded as unused. No deployment is performed.

## Inspected dependency chain

```text
GENAI/data/raw/PCOS_data.csv
  → GENAI saved train/validation/test partitions and split metadata
  → DEEP_LEARNING fitted preprocessing + 378/81/82 matrices
  → existing Phase 3.3 training checkpoint/history
  → existing Phase 3.4 frozen-model evaluation report
  → MLOPS lineage/tracking/gate/registry artifacts
```

The current DL artifacts provide training and validation loss evidence, and the Phase 3.4 test report provides accuracy, precision, recall, F1, and ROC-AUC. They do not contain train/validation classification metrics or the original wall-clock training timestamp; MLOPS records that limitation rather than deriving or inventing values. The existing evaluation report is ingested as evidence and is not rerun.

## Experiment tracking

`experiment_tracking/runs/<training-run-id>.json` records the existing run ID, ingestion timestamp, data/model/preprocessing versions, architecture and training configuration, sample counts, available train/validation/test metrics, training duration, seed, feature schema, model path/hash, synthetic-data status, and run status. The stable training-run ID is derived from the checkpoint and dataset hashes. Pipeline invocations receive separate report IDs.

Tracking and reports use JSON; pipeline events are appended to `logs/mlops.jsonl`. No MLflow server or database is required. The current project had no tracking service or dependency, so this implementation uses the existing Python/YAML stack.

## Dataset versioning

`data_versioning/datasets/<dataset-version>.json` identifies the raw dataset, authoritative split files, transformed arrays, feature names, target, row/feature counts, source hashes, preprocessing choices, and generation source. The version is a SHA-256 composite of the listed immutable inputs. Available CTGAN/TVAE outputs are noted as `REVIEW_REQUIRED` and `used: false`; they are not included in the real-only model's dataset version inputs.

## Phase 2: quality gate, lifecycle, monitoring, and drift readiness

Phase 2 keeps the existing artifact-only behavior: the pipeline consumes the frozen checkpoint, fitted transformer, and saved Phase 3.4 evaluation report. It does not train, reevaluate, deploy, or change the production pointer.

### Quality gate

`configs/mlops.yaml` carries the approved provisional project-level engineering minimums: accuracy `0.75`, precision `0.70`, recall/sensitivity `0.80`, F1 `0.72`, and ROC-AUC `0.85`. These are not clinical standards and are not externally validated medical acceptance criteria. Recall receives particular attention because the project is screening-oriented; precision provides a false-positive safeguard, F1 requires a combined precision/recall floor, accuracy is above the observed majority-class reference, and ROC-AUC requires useful discrimination. Recorded metric values are read from `DEEP_LEARNING/reports/test_evaluation.json`; no metric is invented or rewritten.

### Model lifecycle and deployment readiness

Tracked versions can move through `TRAINED → VALIDATED → REGISTERED → DEPLOYMENT_READY → DEPLOYED`. Only a quality-gate PASS makes a candidate eligible for registration and the separate deployment-readiness checks. Passing the gate is not clinical validation. Deployment readiness checks the checkpoint and preprocessor exist, their hashes match recorded values, evaluation metrics are valid, the gate passes, and registry metadata matches the version. A ready candidate is still `NOT_DEPLOYED`; actual deployment and automatic deployment remain disabled, and no code changes the production pointer. The registry retains `production_model` as an explicit state.

### Prediction logging

The existing dashboard still calls its same predictor. Its request handler now records an operational JSONL event after success or failure in `logs/predictions/predictions.jsonl`: UTC timestamp, checkpoint-derived model version, outcome, inference duration, and on success predicted class/probability. Failure events record a bounded error category. The logger has no clinical-input parameter and stores no raw feature values or patient identifiers. If the log write fails, the prediction response is preserved.

### Monitoring

The MLOPS pipeline writes `reports/monitoring_latest.json` and includes a monitoring snapshot in its per-run report. It aggregates request counts, success rate, latency range/mean, latest timestamp, model version, and positive/negative prediction distribution. Production accuracy, precision, recall, and F1 are not calculated without ground-truth labels; the report states: “Production performance metrics unavailable: ground-truth labels are not available.”

### Drift readiness

The report tracks feature drift, prediction drift, and performance drift separately with `NORMAL`, `WARNING`, `CRITICAL`, or `INSUFFICIENT_DATA` statuses. No current drift finding is claimed. Feature values are intentionally absent from operational logs, so feature drift remains `INSUFFICIENT_DATA` until a privacy-safe feature summary/reference path is approved. Prediction drift remains insufficient until adequate production observations and an approved reference decision policy exist. Performance drift requires ground-truth labels and a reviewed policy.

### Phase 2 pipeline output

```text
quality gate: PASS (with the configured project-level engineering minimums)
registry: REGISTERED (lifecycle DEPLOYMENT_READY if readiness checks pass)
deployment readiness: READY if all checks pass
deployment: NOT_DEPLOYED
feature/prediction/performance drift: INSUFFICIENT_DATA
```

The statuses above describe the current configuration and evidence; they are not model-performance judgments or a claim that drift occurred.

The model version is derived from the official checkpoint's SHA-256, so a different checkpoint yields a different immutable identifier. A version manifest points to the existing checkpoint instead of copying or overwriting it. The registry catalog accepts a version only when the quality gate passes. Registration does not deploy a model or change the production pointer; this workflow never updates a production model automatically.

The current checkpoint is the only discovered DL model version. Model version metadata and registry state are MLOPS-owned files.

## Quality gate

The configured minimums are accuracy `0.75`, precision `0.70`, recall `0.80`, F1 `0.72`, and ROC-AUC `0.85`. These are provisional project-level engineering policies for this academic project, not clinical standards or externally validated medical criteria. Recall receives particular attention for the screening-oriented objective; precision is a false-positive safeguard, F1 combines precision and recall, accuracy is above the observed majority-class reference, and ROC-AUC sets a discrimination floor. The minimums are evaluated against the existing saved metrics and are not inferred or tuned from a new evaluation.

Unset or invalid minimums return `INSUFFICIENT_CONFIGURATION`; this is not a model failure, but the candidate is ineligible for registration/readiness until minimums are configured. With valid minimums, a metric below its minimum returns `FAIL`; the overall gate returns `PASS` only when all five metrics meet their minima. A passing gate means eligibility for registration and readiness review only. It does not establish clinical validity, clinical safety, or medical deployment readiness. Actual deployment remains disabled, and the production pointer is not changed automatically.

The gate reads existing Phase 3.4 metrics for this already-frozen baseline. It does not tune a model or threshold. The held-out test result is reported as existing evidence only; no new evaluation or test-driven checkpoint selection occurs.

## Pipeline and logging

Run:

```powershell
.venv\Scripts\python.exe MLOPS\run_mlops.py
```

The pipeline makes each stage visible:

1. DATA — verify existing raw, split, and transformed artifacts and record the dataset hash.
2. PREPROCESSING — verify the saved fit-on-training-only preprocessing report and feature schema.
3. TRAINING — ingest the existing Phase 3.3 record; do not train.
4. EVALUATION — ingest the existing Phase 3.4 metrics; do not rerun evaluation.
5. QUALITY_GATE — apply the configured engineering minimums to the existing evaluation artifact.
6. MODEL_REGISTRY — track the version; register only on PASS; do not alter production.
7. DEPLOYMENT_READINESS — verify model, preprocessing, hashes, metrics, gate, and registry metadata.
8. MONITORING/DRIFT_READINESS — aggregate operational logs and report evidence sufficiency.
9. DEPLOYMENT — remain `NOT_DEPLOYED`.

Each invocation writes `reports/<pipeline-run-id>.json` and `.md`. A gate failure is a successful MLOps pipeline execution with a rejected registration decision; an artifact or execution error produces a failed report and structured error event.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s MLOPS\tests -v
```

Tests cover missing/pass/failing quality-gate cases, candidate registration protection, deployment readiness, privacy-safe prediction logging, monitoring aggregation, and insufficient drift data. The tests do not train or modify GENAI/Deep Learning artifacts.

## Not included in this foundation

This phase intentionally does not add DVC, MLflow, Prefect, FastAPI, Docker, GitHub Actions, cloud deployment, or automated retraining. Monitoring here is local JSONL aggregation and drift readiness, not a drift detector with unapproved thresholds or an automated model replacement system.
