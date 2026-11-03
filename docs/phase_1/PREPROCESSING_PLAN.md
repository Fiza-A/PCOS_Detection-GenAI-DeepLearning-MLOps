# Preprocessing plan (design only)

## Raw dataset vs modeling dataset

`GENAI/data/raw/PCOS_data.csv` is the immutable source. A future modeling dataset will be derived from training data after provenance decisions; this phase creates no transformed modeling table.

## Proposed decisions for later approval

- Preserve all source columns and values in the raw file. Treat `Sl. No` and `Patient File No.` as `IDENTIFIER_CANDIDATE`; exclude from model features only after documenting that decision. Treat `Unnamed: 44` as an export-artifact candidate pending source clarification.
- Audit and parse numeric-looking strings explicitly (including `a` and `1.99.`); record every coercion. Do not repair suspicious values or interpret zero as missing without evidence.
- If missingness remains, fit numeric and categorical imputation values on training data only and persist them. Missingness indicators may be evaluated and documented.
- Encode known binary fields as categorical only if their source semantics are confirmed. Preserve unknown category codes as categories; obtain a codebook before assigning meanings. Fit encoders on training data and define unknown-category behavior.
- Consider scaling numerical inputs for neural networks using training-only fitted parameters. Preserve the original units in audit/report artifacts. Do not clip IQR outliers by default.
- Retain the target exactly as supplied (`PCOS (Y/N)` values 0/1 observed); documented encoding semantics are not yet established by supplied provenance.
- Review derived/redundant fields (BMI, waist:hip ratio, FSH/LH) without automatically dropping them. Resolve suspicious measurements using source documentation rather than assumed clinical ranges.
- For synthetic tabular generators, retain explicit categorical metadata and ensure generators see only training rows; validate generated values against observed schema and report anomalies without silent correction.

## Split and leakage prevention

Proposed stratified 70/15/15 train/validation/test split with seed 42. Split raw rows before fitting imputers, encoders, scalers, feature selection, or generators. Persist row indices and source-row mapping. Validation is for model selection; test remains untouched until final evaluation. The split has not been created in Phase 1.

Potential target proxies and the diagnosis timing are unresolved because label provenance is unavailable. Resolve provenance before interpreting predictive performance.
