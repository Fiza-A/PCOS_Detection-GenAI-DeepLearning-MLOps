# Phase 2.1 — GENAI diagnostic analysis

## 1. Purpose

This is read-only analysis of saved Phase 2 artifacts. No raw, split, model, synthetic dataset, Phase 2 report, or configuration was changed. No generator was fitted and no data was generated. New code and reports are confined to `GENAI/src/diagnose_phase2.py`, `GENAI/tests/test_phase21_diagnostics.py`, and `GENAI/reports/diagnostics/`.

## 2. Existing Phase 2 status

CTGAN and TVAE both passed the hard technical checks, but both remain `REVIEW_REQUIRED`. No output is approved for Deep Learning. No evidence threshold has been invented in this diagnosis.

## 3. Lineage verification

The raw file has 541 rows. Split metadata and saved index rows both report 378 train, 81 validation, and 82 test. Saved source row indices are disjoint: train∩validation=0, train∩test=0, validation∩test=0. They cover 541 unique source rows. The source SHA-256 matches the split metadata.

For both saved synthesizers, run metadata says seed 42, 378 training rows, 378 generated rows, SDV 1.38.5, 300 epochs, batch 100, CUDA disabled, and the same shared 42-column metadata schema. CTGAN records `pac=10`; TVAE does not use that CTGAN parameter. Model files exist and load as `CTGANSynthesizer` and `TVAESynthesizer`; synthetic CSVs each have 378 rows. Both run records’ training-partition SHA-256 matches the saved `train.csv`; raw file hashes also match. Schema excludes both identifier candidates.

**Training-row use limitation:** the saved metadata says `GENAI/data/processed/train.csv only` and records `validation_or_test_passed_to_fit=false`. These claims are consistent with the recorded training partition hash and implementation. A serialized SDV model does not independently encode the identities of fit rows, so actual fit-row lineage is **NOT VERIFIABLE FROM SAVED ARTIFACTS** beyond the metadata/hash evidence. The artifacts do not provide independent proof that validation/test rows were not passed to `fit`.

## 4. Numerical diagnostics

The existing Phase 2 statistic was recomputed for all 31 numerical features from training and synthetic tables: two-sample KS statistic and similarity `1 − KS`. Recomputed values match the saved per-feature values to floating point precision. Full table: `GENAI/reports/diagnostics/numerical_feature_diagnostics.csv`.

### CTGAN weakest 10

| Feature | Similarity | KS | TVAE similarity |
|---|---:|---:|---:|
| FSH(mIU/mL) | 0.1058 | 0.8942 | 0.7963 |
| Vit D3 (ng/mL) | 0.2169 | 0.7831 | 0.8836 |
| `  I   beta-HCG(mIU/mL)` | 0.4339 | 0.5661 | 0.7434 |
| PRG(ng/mL) | 0.4683 | 0.5317 | 0.8254 |
| `II    beta-HCG(mIU/mL)` | 0.4921 | 0.5079 | 0.6005 |
| Hb(g/dl) | 0.5132 | 0.4868 | 0.8783 |
| LH(mIU/mL) | 0.5794 | 0.4206 | 0.8942 |
| FSH/LH | 0.6085 | 0.3915 | 0.8942 |
| PRL(ng/mL) | 0.6190 | 0.3810 | 0.6614 |
| BP _Systolic (mmHg) | 0.6376 | 0.3624 | 0.9762 |

### TVAE weakest 10

| Feature | Similarity | KS | CTGAN similarity |
|---|---:|---:|---:|
| `II    beta-HCG(mIU/mL)` | 0.6005 | 0.3995 | 0.4921 |
| RBS(mg/dl) | 0.6376 | 0.3624 | 0.6693 |
| PRL(ng/mL) | 0.6614 | 0.3386 | 0.6190 |
| Pulse rate(bpm) | 0.6905 | 0.3095 | 0.8545 |
| `  I   beta-HCG(mIU/mL)` | 0.7434 | 0.2566 | 0.4339 |
| Avg. F size (L) (mm) | 0.7513 | 0.2487 | 0.7540 |
| AMH(ng/mL) | 0.7593 | 0.2407 | 0.8519 |
| TSH (mIU/L) | 0.7778 | 0.2222 | 0.6693 |
| Cycle length(days) | 0.7831 | 0.2169 | 0.8757 |
| BP _Diastolic (mmHg) | 0.7857 | 0.2143 | 0.7593 |

Largest cross-generator numerical disagreements are FSH (CTGAN 0.1058 vs TVAE 0.7963; difference 0.6905), vitamin D3 (0.2169 vs 0.8836; 0.6667), Hb (0.5132 vs 0.8783; 0.3651), progesterone (0.4683 vs 0.8254; 0.3571), and systolic BP (0.6376 vs 0.9762; 0.3386).

### Distribution inspection

The distribution table includes min/max, mean, median, standard deviation, unique count, and 1/5/25/50/75/95/99th percentiles for real train, CTGAN, and TVAE for the union of both generators’ weakest 10. Selected measured patterns:

- **FSH:** training mean 18.79, median 4.885, SD 259.62, max 5052; CTGAN mean 1.57, median 0.21, SD 5.31, max 39.92; TVAE mean 5.63, median 5.26, SD 4.10, max 17.17. The training distribution has an extreme upper tail. Both generators produce much narrower support; CTGAN also piles its lower quartile and median at 0.21.
- **Vitamin D3:** training mean 43.49, median 26.05, SD 277.50, max 5418.6; CTGAN mean 10.14, median 6.077, SD 9.51, max 60.16; TVAE mean 29.66, median 29.18, SD 11.81, max 60.44. TVAE matches central quantiles more closely; both generated ranges are far narrower than the training range.
- **Beta-HCG I:** training mean 487.84, median 12.185, SD 2869.67, max 30007; CTGAN mean 7708.81, median 745.52, SD 11871.21, max 30007; TVAE mean 145.01, median 120.83, SD 136.56, max 671.78. The two generators differ in both center and tail behavior.
- **Beta-HCG II:** training mean 176.13, median 1.99, SD 1323.05, max 25000; CTGAN median 0.99 and max 1439.46; TVAE median 14.74 and max 849.43. Neither reproduces the training extreme tail.
- **RBS:** training median 100, SD 15.71; CTGAN median 90.6, SD 15.50; TVAE median 92.2, SD 7.86. TVAE shows substantially narrower dispersion than the training data.

These are statistical shape descriptions only. No clinical interpretations or expected-range judgments are made. Full values and quantiles are in `numerical_distribution_diagnostics.csv`.

## 5. Categorical diagnostics

Category frequencies are compared across all 11 categorical columns, including the target. CTGAN generated every observed training category for these fields. TVAE omitted Blood Group codes `12` (6 training rows), `14` (12), `16` (8), `18` (1), and Cycle(R/I) code `5` (1). No newly introduced categories were observed. The codes’ meanings are not established by project sources.

Average absolute frequency distortion across observed category levels highlights CTGAN’s largest fields as Fast food (9.26 percentage points), hair growth (8.99), skin darkening (6.88), Cycle(R/I) (5.64), and exercise (2.65). TVAE’s largest are exercise (22.22 pp), hair growth (15.61), skin darkening (13.49), Blood Group (9.06), and Cycle(R/I) (8.64). TVAE’s Blood Group distribution has 4 of 8 observed levels; exercise still has both categories but a strong frequency shift. CTGAN’s largest categorical shifts are concentrated in common binary marginals rather than category loss. Full frequencies and differences: `categorical_feature_diagnostics.csv`.

## 6. Target distribution

| Data | Class 0 | Class 0 % | Class 1 | Class 1 % |
|---|---:|---:|---:|---:|
| Real train | 254 | 67.20% | 124 | 32.80% |
| CTGAN | 217 | 57.41% | 161 | 42.59% |
| TVAE | 267 | 70.63% | 111 | 29.37% |

CTGAN shifts 37 records from class 0 to class 1 relative to the real training counts: class-0 count −14.57%, class-1 count +29.84%, each class share differs by 9.79 percentage points. TVAE has class-0 count +5.12% and class-1 count −10.48%; each class share differs by 3.44 pp. These are descriptive differences, not an arbitrary pass/fail threshold. See `target_distribution_diagnostic.csv`.

## 7. Association diagnostics

The stored Phase 2 matrices are compared directly using the Phase 2 Pearson/NMI/correlation-ratio measures. Largest absolute errors:

- Both models struggle with FSH vs FSH/LH: real 0.9741, CTGAN −0.0307 (error 1.0048), TVAE −0.0032 (error 0.9773).
- CTGAN also has large differences for Weight–BMI (real 0.9116, CTGAN −0.0165, error 0.9281), Hip–Waist (0.8742 vs −0.0069, error 0.8812), and left/right follicle counts (0.7933 vs −0.0147, error 0.8081).
- TVAE is closer on several high-association pairs: BMI–Hip 0.6037 vs 0.6103 (error 0.0065), Weight–Hip 0.6351 vs 0.6636 (0.0286), Weight–Waist 0.6673 vs 0.5969 (0.0704), and BMI–Waist 0.6368 vs 0.5892 (0.0477). It still misses other strong associations, including FSH/FSH-LH and the bilateral follicle-count relationship (real 0.7933, TVAE 0.5853, error 0.2080).
- TVAE creates some large associations where training values are weak, such as pimples–fast food (real 0.0768, TVAE 0.5635) and hair growth–skin darkening (0.0958 vs 0.5412).

Association values are statistical and do not imply clinical causality. Full pairwise values: `association_diagnostic.csv`.

## 8. Diversity diagnostics

Both synthetic datasets have 378 unique full rows, zero exact duplicate rows, and zero exact-record matches to train/validation/test. Categorical pattern evidence differs: across the 10 non-target categorical columns CTGAN has 305 distinct patterns, 73 repeated rows, and a maximum pattern frequency of 4. TVAE has 87 distinct patterns, 291 repeated rows, and one categorical pattern repeated 85 times. Thus zero full-row duplicates do not imply comparable categorical diversity; TVAE shows substantial category-combination concentration.

## 9. Nearest-neighbour and exact-match diagnostics

Distances use the existing Gower-like metric: numeric differences scaled by training ranges, plus categorical mismatch. Training leave-one-out nearest distances have min 0.0352, Q1 0.0699, median 0.0856, Q3 0.0989, Q95 0.1245, max 0.2023.

| Generator | Min | Q1 | Median | Q3 | Max | Below train LOO Q1 | Below train LOO median | Below train LOO Q3 | Below train LOO Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTGAN | 0.0588 | 0.1109 | 0.1269 | 0.1408 | 0.1863 | 0.26% | 3.70% | 11.64% | 47.09% |
| TVAE | 0.0281 | 0.0527 | 0.0627 | 0.0739 | 0.1048 | 67.20% | 91.01% | 98.15% | 100.00% |

TVAE’s lower median reflects a broad shift to closer training neighbours, not only a few extreme cases: 91.01% are below the training leave-one-out median, and all are below its Q95. Five TVAE records are below the training leave-one-out Q1 (0.01) reference; its minimum is below the smallest training leave-one-out distance. CTGAN is farther than the training leave-one-out median for most records; 47.09% are below training LOO Q95.

Closest pairs are listed by synthetic-row index, training-partition row index, and source row index in `nearest_record_pairs.csv`; patient/file IDs are not exposed. Exact full-record overlaps are 0 for CTGAN and TVAE against each of train, validation, and test. These are preliminary indicators only; they do not establish anonymity or a formal privacy guarantee. The distance depends on the selected feature-space scaling and must not be interpreted as a formal attack result.

## 10. Preprocessing diagnosis

### OBSERVED

- IDs are excluded. Categorical fields (including the target) are supplied to SDV with categorical metadata; numerical measurements use numerical metadata. No feature scaling is applied before SDV.
- Numeric-looking text fields are parsed. One unparseable AMH training value was replaced by the training median (3.81); categorical missing entries are filled with training modes. No validation/test values fit these parameters.
- SDV 1.38.5 synthesizer defaults `enforce_min_max_values=True` and `enforce_rounding=True` were not overridden in the run code/config. Generated numeric values in observed reports stay within training support for these features.
- TVAE omits five rare training categories and concentrates 378 records into 87 non-target categorical patterns. CTGAN preserves all observed category levels but has major numeric-tail/association errors.

### SUPPORTED INFERENCE

- Sparse category counts (one to twelve training rows for TVAE’s omitted states) are consistent with difficulty reproducing rare categorical outcomes in a 378-row fit. The missing categories were not deliberately removed by preprocessing.
- Extreme training tails, especially FSH, vitamin D3, and beta-HCG, are consistent with the large KS and range/dispersion differences. Default min/max enforcement can constrain samples to training bounds, but does not ensure that the interior tail distribution is reproduced.
- TVAE’s broad nearest-distance shift and repeated categorical patterns are consistent with its lower categorical diversity and closer synthetic records under this particular distance definition.

### NOT ESTABLISHED

- Whether categorical collapse, tail compression, or association loss was caused by the preprocessing, SDV metadata, model family, the small dataset, or random variation. Only one seed was run; this diagnosis cannot separate those factors.
- Whether the extreme training values are valid measurements or source errors; units and code meanings are **Not established from available project sources.**
- Whether the saved models actually saw only train rows. Metadata and partition hashes are consistent, but serialized model files do not independently prove fit-row identity.

No preprocessing or model setting was changed in this diagnostic phase.

## 11. CTGAN diagnosis

1. **Reproduces well:** all observed category levels in this run; low full-row duplicate and exact-match counts; categorical marginal similarities are generally stronger than TVAE’s. Aggregate association similarity is 0.9429.
2. **Reproduces poorly:** FSH and vitamin D3 central/tail distributions, both beta-HCG tails, and several strong numeric relationships.
3. **Weakness drivers:** FSH similarity 0.1058, vitamin D3 0.2169, FSH/FSH-LH association error 1.0048, Weight/BMI error 0.9281.
4. **Area:** numerical tail fidelity and relationships, plus target/Fast food marginal shifts.
5. **Mode collapse:** no observed categorical level disappearance; FSH/Vit D lower quantiles show concentration at training minimum. Categorical pattern repetition is moderate (305 patterns, maximum 4).
6. **Exact memorization:** no exact full-row overlap observed. This does not prove privacy.
7. **Unresolved:** seed stability, meaningful acceptance criteria, clinical validity, and independent fit-row provenance.

## 12. TVAE diagnosis

1. **Reproduces well:** better aggregate numeric similarity (0.8329), target marginal, and several anthropometric relationships (BMI–Hip/Weight–Hip/Weight–Waist).
2. **Reproduces poorly:** rare categories, exercise frequency, beta-HCG II, RBS/PRL/pulse distributions, and categorical pattern variety.
3. **Weakness drivers:** Blood Group level loss, 85-fold repeated categorical pattern, beta-HCG II similarity 0.6005, and FSH/FSH-LH association error 0.9773.
4. **Area:** categorical diversity and broad nearest-neighbour closeness, with selected numerical-tail and association errors.
5. **Mode collapse:** clear categorical concentration evidence: 4/8 Blood Group codes and 2/3 Cycle codes represented; non-target categorical combinations 87/378, one repeated 85 times.
6. **Exact memorization:** no exact full-row overlap observed. Close-neighbour results remain a preliminary risk signal, not proof of memorization.
7. **Unresolved:** whether lower distances arise from broad mode concentration, learned distribution shape, preprocessing, or distance-space weighting; seed stability and authoritative data definitions.

## 13. Why both remain REVIEW_REQUIRED

| Gate dimension | CTGAN | TVAE |
|---|---|---|
| Technical status | Passed schema, target, dtype, and finite-data checks | Same |
| Numerical fidelity | Mean 1−KS 0.6832; FSH 0.1058; vitamin D3 0.2169 | Mean 0.8329; beta-HCG II 0.6005; RBS 0.6376 |
| Categorical fidelity | Mean similarity 0.9515; target share differs 9.79 pp; several common-field shifts | Mean 0.8752; omitted five rare levels; strong exercise distortion and combination concentration |
| Association | Mean similarity 0.9429, but FSH/FSH-LH and anthropometric links fail | 0.9323; same FSH/FSH-LH failure and spurious categorical relations |
| Target | 0:217/1:161 vs 254/124 training | 0:267/1:111, closer marginally |
| Diversity | 0 exact duplicates, but 73 repeated categorical patterns | 0 exact duplicates, but 291 repeated categorical patterns; max pattern 85 |
| Privacy indicators | No exact overlap; median nearest distance 0.1269 | No exact overlap; broad closer-neighbour shift, median 0.0627 |
| Clinical validity | Not established from available project sources | Same |
| Threshold policy | No justified acceptance thresholds | Same |

The `REVIEW_REQUIRED` state follows from hard technical checks passing while material quality/privacy evidence remains unresolved and project sources define no acceptance thresholds. The higher average on selected metrics is insufficient for approval.

## 14. Limitations

This is one seed on a small dataset. All fidelity, association, and distance measures are descriptive and depend on the metric definitions. No clinical codebook or verified units/ranges are available. Exact non-overlap is not an anonymity guarantee. Fit-row isolation is supported by code/run metadata and matching hashes but is not independently embedded in the model file. No downstream utility experiment has been run.

## 15–16. Recommended next action

### OPTION B — CURRENT EVIDENCE SUFFICIENT; NO RERUN JUSTIFIED

The diagnostic identifies meaningful differences but no single confirmed implementation defect whose one-parameter change has a clear, isolated acceptance criterion. Multiple weaknesses span skewed numeric tails, associations, and rare categorical states; a further run selected to chase current scores would risk post-hoc tuning. Keep the current artifacts `REVIEW_REQUIRED`. A future experiment should first state a prospective question and evaluation protocol; clinical interpretation additionally needs authoritative provenance/codebook information. This recommendation does not mean either generator is approved or that the observed limitations are acceptable.
