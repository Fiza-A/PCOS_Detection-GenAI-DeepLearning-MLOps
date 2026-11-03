# Phase 3.1 — Deep Learning Data Integration and Preprocessing

## Scope and outcome

This phase creates numeric inputs for the future `REAL_ONLY_BASELINE`. It does not train a neural network, evaluate a classifier, run SHAP, create a new split, or use synthetic data. The train-fitted preprocessing pipeline completed and all recorded leakage checks passed.

## Dataset and authoritative artifacts

The raw source is `GENAI/data/raw/PCOS_data.csv`. Its SHA-256 matches the source hash recorded in the authoritative metadata: `4c2b8fddb281239f1cfba6581deb5b429f95a152f63b2e9f06749d7b75928d8d`.

The implementation consumed the existing processed partitions and lineage files directly:

- `GENAI/data/processed/train.csv`
- `GENAI/data/processed/validation.csv`
- `GENAI/data/processed/test.csv`
- `GENAI/data/processed/split_metadata.json`
- `GENAI/data/processed/split_indices.csv`

No split was recalculated. The saved source row indices are disjoint and collectively cover source rows 0–540. Each partition’s target values were checked against the corresponding raw source rows. Counts match the authoritative metadata:

| Partition | Rows | Target 0 | Target 1 |
|---|---:|---:|---:|
| Train | 378 | 254 (67.20%) | 124 (32.80%) |
| Validation | 81 | 55 (67.90%) | 26 (32.10%) |
| Test | 82 | 55 (67.07%) | 27 (32.93%) |
| Total raw data | 541 | 364 | 177 |

The test partition remains isolated from preprocessing fit and is only transformed after the single training fit.

## Target

Target: `PCOS (Y/N)`. The observed raw source encoding is binary `0`/`1`, with counts 364/177. The same encoding is preserved in `y_train.npy`, `y_validation.npy`, and `y_test.npy` as `int64`; no target recoding or balancing was performed.

## Feature handling

The raw dataset has 45 columns. After excluding the target from X, 41 candidate features are modeled: 31 numerical and 10 categorical. The serialized feature inventory at `DEEP_LEARNING/reports/feature_inventory.csv` lists every raw column, original dtype, raw and training-partition missing counts, raw unique count, preprocessing operation, inclusion flag, and exclusion reason.

Identifiers `Sl. No` and `Patient File No.` are excluded from X as identifier candidates. They remain in the raw dataset and source artifacts. `PCOS (Y/N)` is kept separately as y. `Unnamed: 44` is retained in the raw source but excluded because the authoritative Phase 2 configuration lists it as an export candidate; it is not silently discarded or treated as a model feature.

The 31 numerical features are: ` Age (yrs)`, `Weight (Kg)`, `Height(Cm) `, `BMI`, `Pulse rate(bpm) `, `RR (breaths/min)`, `Hb(g/dl)`, `Cycle length(days)`, `Marraige Status (Yrs)`, `No. of abortions`, `  I   beta-HCG(mIU/mL)`, `II    beta-HCG(mIU/mL)`, `FSH(mIU/mL)`, `LH(mIU/mL)`, `FSH/LH`, `Hip(inch)`, `Waist(inch)`, `Waist:Hip Ratio`, `TSH (mIU/L)`, `AMH(ng/mL)`, `PRL(ng/mL)`, `Vit D3 (ng/mL)`, `PRG(ng/mL)`, `RBS(mg/dl)`, `BP _Systolic (mmHg)`, `BP _Diastolic (mmHg)`, `Follicle No. (L)`, `Follicle No. (R)`, `Avg. F size (L) (mm)`, `Avg. F size (R) (mm)`, and `Endometrium (mm)`.

The 10 categorical features are: `Blood Group`, `Cycle(R/I)`, `Pregnant(Y/N)`, `Weight gain(Y/N)`, `hair growth(Y/N)`, `Skin darkening (Y/N)`, `Hair loss(Y/N)`, `Pimples(Y/N)`, `Fast food (Y/N)`, and `Reg.Exercise(Y/N)`. These classifications follow the existing Phase 2 schema; no additional clinical meanings were inferred.

## Missing values and upstream preprocessing

The raw source has one missing value in `Marraige Status (Yrs)` and one in `Fast food (Y/N)`. `Unnamed: 44` has 539 raw missing values and is excluded as documented above. The Phase 2 parse audit also records one non-missing but unparseable source value in `AMH(ng/mL)`.

The saved Phase 2 partitions contain zero missing cells in model features. Phase 2 already applied its training-fitted numeric median and categorical mode handling, including parsing the specified numeric-text fields. The DL pipeline therefore receives those existing imputed values; it does not recreate or alter the partitions. A DL `SimpleImputer` is included in the reusable pipeline and is fitted only on train rows. Given the already-imputed inputs, no further values required imputation in this phase.

## Preprocessing pipeline

One sklearn `ColumnTransformer` was fitted once on `train.csv` model features only:

- Numerical branch: `SimpleImputer(strategy="median")` followed by `StandardScaler`.
- Categorical branch: `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`.
- Validation and test data were passed only to `transform`; their values did not affect fitted medians, scaling parameters, or encoder categories.
- Categories absent from the training partition are ignored by the encoder and do not add or reorder columns.
- Transformed matrices are saved as finite `float32`; labels are saved as `int64`.

The fitted, reusable pipeline is `DEEP_LEARNING/models/preprocessing_pipeline.joblib`. Encoded column names and their order are recorded in `DEEP_LEARNING/reports/transformed_feature_names.json`.

## Final representation

| Array | Shape |
|---|---:|
| `X_train.npy` | 378 × 58 |
| `X_validation.npy` | 81 × 58 |
| `X_test.npy` | 82 × 58 |
| `y_train.npy` | 378 |
| `y_validation.npy` | 81 |
| `y_test.npy` | 82 |

All arrays are under `DEEP_LEARNING/data/processed/`. The 41 source feature columns become 58 numeric columns after one-hot encoding. No rows were added, removed, rebalanced, or synthesized.

## Leakage checks

All checks passed:

- Test and validation source row indices are disjoint from training indices.
- The three partitions are pairwise disjoint and collectively exhaustive.
- Preprocessing is fitted on the training partition only.
- Target is not present in X.
- Both identifier candidates are excluded from X.
- No NaN or infinite values remain in transformed matrices.
- All partitions share the fitted 58-column feature schema and order.

## Tests and reproducibility

Command: `.venv\Scripts\python.exe -m unittest discover -s DEEP_LEARNING\tests -v`

Result: **13 tests passed**. Tests cover partition sizes and indices, source target encoding, identifier/target exclusion, finite transformed values, feature dimensions/order, train-only imputer and encoder fitting, repeatable outputs, and raw source integrity.

The preprocessing was executed twice in isolated test output directories. The saved X/y arrays and transformed feature names matched exactly. `random_seed = 42` is recorded from the authoritative split; preprocessing itself is deterministic and does not shuffle or draw random values.

## Files produced

- `DEEP_LEARNING/src/preprocess_data.py`
- `DEEP_LEARNING/tests/test_preprocessing.py`
- `DEEP_LEARNING/models/preprocessing_pipeline.joblib`
- `DEEP_LEARNING/data/processed/X_{train,validation,test}.npy`
- `DEEP_LEARNING/data/processed/y_{train,validation,test}.npy`
- `DEEP_LEARNING/reports/feature_inventory.csv`
- `DEEP_LEARNING/reports/transformed_feature_names.json`
- `DEEP_LEARNING/reports/preprocessing_report.json`
- `docs/phase_3/PHASE_3_1_PREPROCESSING.md`

## Restrictions observed

No CTGAN/TVAE data was loaded or used. No GENAI files or artifacts were modified. No neural network, classifier, accuracy/AUC evaluation, SHAP explanation, or hyperparameter selection was run. The work stops at data integration and preprocessing.

## Status

`PHASE 3.1 COMPLETE`
