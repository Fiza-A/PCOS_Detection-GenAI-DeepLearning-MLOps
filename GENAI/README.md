# GENAI — PCOS synthetic tabular data

GENAI is independently runnable and owns the Phase 2 CTGAN/TVAE experiment. It reads the immutable source at `data/raw/PCOS_data.csv`, creates a stratified train/validation/test split, fits preprocessing parameters on training rows only, and gives the two SDV synthesizers only `data/processed/train.csv`. Validation and test are excluded from both generator fits.

## Run

From the project root with the pinned environment installed:

```powershell
.venv\Scripts\python.exe GENAI/run_genai.py
```

## Method

- Split: stratified 70/15/15 by `PCOS (Y/N)`, seed 42.
- `Sl. No` and `Patient File No.` stay in the raw CSV but are excluded from modeling and synthetic schema. `Unnamed: 44` is excluded as the audited mostly-empty export artifact.
- Numeric-looking string fields are parsed explicitly. Unparseable/missing numerical values use medians learned from training rows. Categorical missing values use training modes; coded categories remain strings. No medical ranges or category meanings are invented.
- Shared SDV metadata makes categorical codes and target categorical and measurement fields numerical. Both models use the same schema, training partition, seed, epoch count, batch size, and synthetic sample count.
- Reports cover per-feature KS and Jensen-Shannon similarity, pairwise association, technical schema/support checks, diversity, exact overlap, preliminary nearest-neighbour privacy indicators, and a conservative review gate.

## Outputs

- `data/processed/{train,validation,test}.csv`, `split_indices.csv`, and `split_metadata.json`
- `models/shared/pcos_metadata.json`, model files in `models/{ctgan,tvae}/`
- synthetic samples and run metadata in `data/synthetic/{ctgan,tvae}/`
- validation and evaluation reports in `reports/{data_validation,ctgan,tvae,comparison,quality_gate}/`
- experiment narrative: `../docs/phase_2/PHASE_2_GENAI.md`

Quality metrics are descriptive and do not establish clinical validity, privacy, anonymity, or downstream utility. With no justified acceptance thresholds in project sources, a technically valid generator remains `REVIEW_REQUIRED` rather than automatically approved.
