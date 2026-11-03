# Phase 2 — GENAI generation and evaluation

## 1. Objective

Evaluate fresh SDV CTGAN and TVAE fits on the actual PCOS training partition. This phase assesses statistical fidelity, associations, technical structure, diversity, and preliminary privacy-risk indicators. It does not implement or assess downstream classifier utility.

## 2. Dataset

Raw source is the byte-preserved `GENAI/data/raw/PCOS_data.csv` (541 × 45; SHA-256 recorded in `data/processed/split_metadata.json`). The project materials do not establish the original publisher/collection context, code meanings, verified units/ranges, or label-generation method. **Not established from available project sources.**

## 3. Split

Seed 42; two-stage stratified split on `PCOS (Y/N)` gives 378 train, 81 validation, and 82 test rows. Class counts:

| Partition | Total | 0 | 1 |
|---|---:|---:|---:|
| Train | 378 | 254 | 124 |
| Validation | 81 | 55 | 26 |
| Test | 82 | 55 | 27 |

Source row indices are saved in `GENAI/data/processed/split_indices.csv` and full deterministic metadata in `split_metadata.json`. Both synthesizer fit calls load only `train.csv`. Validation/test are not generator training data or the primary fidelity reference. Their only use here is a supplementary exact-record/nearest-neighbor risk comparison after synthesis.

## 4. Preprocessing

Only training rows determine numeric medians and categorical modes. Two numeric-looking text fields are parsed as numeric; a terminal period is removed as formatting, and unparseable values become missing before train-median imputation. Categorical missing entries are filled using training modes; codes remain categorical strings. The observed target values 0/1 are retained as categorical labels. No medical values are clipped or medically reinterpreted. `Sl. No` and `Patient File No.` remain in the raw dataset but are omitted from partition model features. `Unnamed: 44` is omitted as an audited mostly-empty export field. The parse audit and fill parameters are in `split_metadata.json`.

## 5. Generator schema

Shared SDV metadata is saved to `GENAI/models/shared/pcos_metadata.json` and validated before fitting. It has 42 columns: 31 numerical predictors, 10 categorical predictors, and categorical target `PCOS (Y/N)`. It excludes two identifiers and the export artifact. Missing values are resolved by the training-fitted imputation described above. Both generators use this same schema.

## 6–8. CTGAN and TVAE configuration/training

Both use SDV 1.38.5, PyTorch 2.15.0+cpu, seed 42, batch size 100, and 300 epochs. CTGAN uses `pac=10`; TVAE does not use that CTGAN-specific option. GPU is disabled. Each model was fit only on the same 378 training rows and generated 378 records. Model files and per-run config, version, timestamps, and data hashes are saved under `GENAI/models/` and `GENAI/data/synthetic/{ctgan,tvae}/`.

## 9. Synthetic sample size

The configured sample count is 378, equal to the actual training partition size. Both use the same count.

## 10. Fidelity methodology and actual results

For numerical features, report two-sample KS statistic and `1 − KS`; this is a descriptive similarity score, not a hypothesis-testing conclusion. For categorical features, report base-2 Jensen-Shannon distance and `1 − distance`. Feature results and best/worst lists are in `reports/{ctgan,tvae}/fidelity.json`.

| Metric | CTGAN | TVAE |
|---|---:|---:|
| Mean numerical 1−KS | 0.6832 | 0.8329 |
| Mean categorical JS similarity | 0.9515 | 0.8752 |
| Target JS similarity | 0.9141 | 0.9684 |

CTGAN’s weakest numerical similarities were FSH (0.1058), vitamin D3 (0.2169), beta-HCG I (0.4339), progesterone (0.4683), and beta-HCG II (0.4921). TVAE’s weakest were beta-HCG II (0.6005), RBS (0.6376), prolactin (0.6614), pulse (0.6905), and beta-HCG I (0.7434). Both have feature-level failures that aggregate means conceal.

CTGAN generated target counts 0:217 / 1:161; TVAE 0:267 / 1:111, compared with training 0:254 / 1:124. TVAE’s target marginal was closer by the JS measure, while that alone is not an approval criterion.

## 11. Correlation/association methodology

Numerical/numerical pairs use Pearson correlation; categorical/categorical pairs use normalized mutual information; mixed pairs use correlation ratio eta-squared. We compare absolute pairwise association values, average absolute error, and `1 − error`. These are association summaries, not clinical causality.

Aggregate association similarity: CTGAN **0.9429**, TVAE **0.9323**. Largest mismatch for both was FSH vs FSH/LH: real-training association 0.9741, synthetic −0.0307 (CTGAN) and −0.0032 (TVAE). Other large mismatches are recorded in each generator’s `correlation.json` and association matrices. Derived/redundant feature relationships are not interpreted as causal.

## 12. Diversity methodology

Both synthetic tables have 378 unique full rows, 0 duplicate rows, and 0 duplicate rate. Category-combination repetition, uniqueness, and category disappearance are in `diversity_privacy.json` and `data_validation/*.json`. Zero duplicate rate is not treated as sufficient evidence of diversity. TVAE omitted several rare training levels (Blood Group codes 12, 14, 16, 18 and Cycle code 5); CTGAN represented all observed categorical training levels in this sample.

## 13. Preliminary privacy-risk methodology

Exact full-record overlap was checked against training, validation, and test after synthesis. Both had zero exact overlaps in each partition. A Gower-like distance combines numeric absolute difference scaled by training range and categorical mismatches. Median nearest-training distance was 0.1269 for CTGAN and 0.0627 for TVAE; training leave-one-out median was 0.0856. TVAE’s closer nearest records warrant review, but these descriptive distances do not constitute an attack evaluation or formal privacy guarantee. No anonymity or differential-privacy claim is made.

## 14. Technical validity and clinical limits

Both generated tables passed implemented hard technical checks: expected columns/order, compatible data types, target support, no included identifiers, no missing cells, and finite numeric output. No synthetic value was labeled medically impossible. Clinical plausibility and authoritative units/ranges are **Not established from available project sources.** Out-of-training-support checks are reported as statistical support differences only.

## 15. Quality-gate methodology

Hard schema/type/target/data-integrity checks can reject technically invalid output. Fidelity, relationship preservation, distributional shifts, diversity, and preliminary privacy indicators remain review criteria because no scientifically justified acceptance thresholds are documented. Both models therefore have status **REVIEW_REQUIRED**, `approved_for_downstream_use=false`. Neither is APPROVED; no threshold was invented to force a pass.

## 16–19. Results and comparison

| Dimension | CTGAN | TVAE | Measured reading |
|---|---:|---:|---|
| Hard technical checks | PASS | PASS | Both schema-compatible in this run |
| Numerical mean 1−KS | 0.6832 | 0.8329 | TVAE higher aggregate; both have weak features |
| Categorical mean JS similarity | 0.9515 | 0.8752 | CTGAN higher; TVAE drops rare levels |
| Association similarity | 0.9429 | 0.9323 | CTGAN slightly higher aggregate; notable shared mismatch |
| Target counts | 217 / 161 | 267 / 111 | Real train 254 / 124 |
| Duplicate rate | 0 | 0 | Full-row uniqueness only |
| Exact overlap with train/val/test | 0 / 0 / 0 | 0 / 0 / 0 | No exact matches observed |
| Median nearest-training distance | 0.1269 | 0.0627 | TVAE records are closer by this metric |
| Gate | REVIEW_REQUIRED | REVIEW_REQUIRED | Neither approved |

CTGAN is stronger on aggregate categorical and association summaries. TVAE is stronger on mean numerical and target-marginal similarity, but collapses rare categories and has closer nearest-training records. This evidence does not establish a universally preferable generator. Neither synthetic table is approved for downstream use on the current policy/evidence.

## 20. Limitations

The dataset and training partition are small; fidelity is seed- and sample-dependent. Only one seed was run. IQR/support and distance measures are descriptive. There is no formal privacy guarantee, clinical review, authoritative data dictionary, or downstream predictive utility measurement. Very skewed measurements and derived feature relationships are difficult for both generators. Clinical validity remains unestablished.

## 21. Final decision

Both models completed successfully and passed technical checks, but both remain **REVIEW_REQUIRED**. No generator is selected or approved. The current artifacts must not be treated as approved synthetic inputs for Deep Learning until a justified acceptance policy and further review are established.

## 22. Reproducibility

Use `.venv` and pinned root dependencies, then run `.venv\Scripts\python.exe GENAI\run_genai.py` from the project root. Configuration is in `GENAI/configs/phase2.yaml`; seed 42, 300 epochs, batch 100, and sample count 378 were used. Split lineage and generator-specific run metadata record hashes and versions. Run logs are in `GENAI/reports/phase2_run.log`. **Synthetic data was generated only from the training partition.**
