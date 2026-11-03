# PCOS-MLOps

Academic project organized into three independent subjects: **GENAI → DEEP_LEARNING → MLOPS**. GENAI owns the CTGAN/TVAE experiments, DEEP_LEARNING owns the real-only DLNN baseline, and MLOPS now tracks and versions the existing run through a configurable quality gate and local registry.

The raw source is kept at `GENAI/data/raw/PCOS_data.csv`. Phase 2 generator results and limitations are documented in `docs/phase_2/PHASE_2_GENAI.md`. Both generators currently remain `REVIEW_REQUIRED`; neither dataset is approved for downstream consumption. The current MLOPS run ingests existing DL artifacts only; its quality-gate minima are intentionally unset, so the model is not registered or deployed.
