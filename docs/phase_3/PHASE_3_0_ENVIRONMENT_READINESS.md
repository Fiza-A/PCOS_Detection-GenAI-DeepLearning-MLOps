# Phase 3.0 — Deep Learning Environment Readiness

## Result

**READY** — the project-local environment imports PyTorch, scikit-learn, matplotlib, and SHAP successfully. CPU PyTorch functionality and a basic matplotlib figure were verified, and `pip check` reports no broken requirements. No model was trained and no SHAP explainer was created.

## Project and data readiness

- Project root: `1PCOS_SEM7`.
- `DEEP_LEARNING/` exists and remains reserved; it contains only its existing README.
- Authoritative split artifacts are under `GENAI/data/processed/`:
  - `train.csv`: 378 rows
  - `validation.csv`: 81 rows
  - `test.csv`: 82 rows
  - `split_metadata.json` and `split_indices.csv`
- Total rows: 541. The split metadata reports seed 42, stratified on `PCOS (Y/N)`.
- Partition target counts match metadata: train 254/124, validation 55/26, test 55/27 for target categories 0/1.
- Verified that the row-index partitions are pairwise disjoint and their union covers all source row indices 0–540. The three CSV row counts match the authoritative metadata.
- No second split was created. No preprocessing object was fitted.
- `split_metadata.json` records that generator fitting used `train.csv` only and validation/test rows were not passed to either synthesizer. The DL baseline can read the three real processed partitions while retaining that source-row lineage; the test partition must remain held out through model selection.

## Environment

Verified using `.venv\Scripts\python.exe`:

| Component | Result |
|---|---|
| Python | 3.11.15 |
| Operating system / architecture | Windows 10, AMD64 |
| PyTorch | 2.15.0+cpu; imports successfully |
| CUDA | Unavailable; not required |
| PyTorch tensor operation | CPU tensor and `torch.nn.Linear` forward pass succeeded |
| `torch.nn` / `torch.optim` | Imports succeeded |
| pandas | 2.3.3 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| scikit-learn model selection, metrics, calibration, preprocessing | Required imports succeeded, including ROC-AUC, average precision (PR-AUC), confusion matrix, calibration utilities, and preprocessing utilities |
| Python/NumPy/PyTorch seed support | Repeated seeded random draws matched in the environment check |
| matplotlib | 3.11.2; imports successfully and a basic `Figure` object was created (not displayed or saved) |
| SHAP | **SHAP_STATUS = AVAILABLE** — 0.51.0; imports successfully; no explainer or explanation was created |
| pip dependency integrity | `pip check`: `No broken requirements found.` (exit code 0) |

Only matplotlib and SHAP were requested for installation. Their required supporting packages were added by pip. Python, PyTorch, NumPy, pandas, and scikit-learn remained at their audited versions: Python 3.11.15, PyTorch 2.15.0+cpu, NumPy 2.4.6, pandas 2.3.3, and scikit-learn 1.9.1.

## Synthetic-data restriction

The first DL experiment is `REAL_ONLY_BASELINE`. It may use only the real training partition for fitting, real validation partition for model selection, and untouched real test partition for final evaluation. CTGAN and TVAE synthetic rows must not be loaded into this baseline. Synthetic augmentation experiments remain blocked pending an explicit later decision; neither generator is approved for downstream use.

## Tests

No `DEEP_LEARNING/tests/` directory exists, so no DL tests were run. No model tests were created. No model was trained.

## Changes and preserved state

Only this readiness report was updated. No files under `GENAI/` were changed. No split, dataset, preprocessing artifact, or generator output was modified. No project requirements or source files were changed.

## Final status

`PHASE 3.0 READY`

The required packages import successfully and the package environment is consistent. This readiness task did not proceed to Phase 3.1 or train a model.
