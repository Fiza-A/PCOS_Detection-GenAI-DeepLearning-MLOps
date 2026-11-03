# Column inventory

Machine-readable inventory: `GENAI/data/processed/column_inventory.csv`. The audit utility reads the source without changing it and emits original name, metadata-only normalized name, dtype, inferred semantic type, missing count/percentage/representation, unique count, sample values, numeric descriptive statistics, zero/negative counts, and a low-confidence note.

All 45 source columns are represented in original order. Names are not normalized in the raw dataset. Semantic typing is a mechanical preliminary inference from dtype/cardinality/name, not a verified codebook. Numeric-looking object fields are marked as numerical-stored-as-text. The target is marked separately. The two sequential-looking columns are `IDENTIFIER_CANDIDATE`, not removed.

Category frequencies for inferred categorical and binary fields are in `GENAI/data/processed/data_quality_report.json`. See `DATA_QUALITY_REPORT.md` for measured issues and interpretation limits.
