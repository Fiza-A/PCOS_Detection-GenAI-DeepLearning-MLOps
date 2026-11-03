# Faculty Execution Output

## Purpose

The GENAI entry point now presents the existing Phase 2 run in clearly numbered sections. It displays the dataset and split sizes, generator training and sampling status, elapsed stage times, measured fidelity and diversity indicators, the actual gate decisions, and a short interpretation of `REVIEW_REQUIRED`. ASCII punctuation is used in the banner and gate messages for consistent rendering in default Windows consoles.

## Files changed

- `GENAI/run_genai.py` — presentation-only console output and elapsed-time display around the existing generator calls.
- `docs/phase_2/FACULTY_EXECUTION_OUTPUT.md` — this verification record.

No generator, preprocessing, split, evaluation, comparison, or quality-gate module was modified. No tests were changed.

## Scientific behavior and data integrity

The existing experiment configuration and scientific behavior were preserved. CTGAN and TVAE continue to use the existing training-only fit contract and existing split, seed, configuration, sample count, metrics, and gate policy. Validation and test data were not passed to generator fitting. The quality gate remains unchanged.

The raw source file SHA-256 before and after execution was:

```text
4C2B8FDDB281239F1CFBA6581DEB5B429F95A152F63B2E9F06749D7B75928D8D
```

The raw dataset remained byte-for-byte unchanged. Existing artifacts and diagnostic evidence were preserved; the requested real pipeline and diagnostic commands ran using their existing output behavior.

## Verification results

- Tests: `python -m unittest discover -s GENAI\tests -v` using the project `.venv`; 12 tests passed.
- Main execution: `.venv\Scripts\python.exe GENAI\run_genai.py` completed successfully with exit code 0.
- Diagnostics: `.venv\Scripts\python.exe GENAI\src\diagnose_phase2.py` completed successfully with exit code 0.
- Final gate results: CTGAN `REVIEW_REQUIRED`; TVAE `REVIEW_REQUIRED`.

## Observed run output

The verified run read 541 records and 45 columns. Its existing stratified split contained 378 training, 81 validation, and 82 test rows (seed 42). Each generator produced 378 rows.

| Measured item | CTGAN | TVAE |
|---|---:|---:|
| Numerical fidelity (`1 - KS`) | 0.6832 | 0.8329 |
| Categorical fidelity (JS similarity) | 0.9515 | 0.8752 |
| Association fidelity | 0.9429 | 0.9323 |
| Duplicate rows | 0 | 0 |
| Exact overlap with training | 0 | 0 |
| Quality gate | REVIEW_REQUIRED | REVIEW_REQUIRED |

The displayed run reported CTGAN stage time of 80.5 seconds and TVAE stage time of 38.1 seconds. Stage time covers each existing train function, including fitting, sampling, and artifact saving. The console also calculates the total pipeline time from the current run; timings are run-specific and are not configuration values.

## Example of the faculty-facing message

```text
PCOS-MLOps - GENERATIVE AI EXECUTION
[1/6] DATASET & EXPERIMENT SETUP
[2/6] DATA PARTITIONING
[3/6] CTGAN TRAINING
[4/6] TVAE TRAINING
[5/6] SYNTHETIC DATA EVALUATION
[6/6] QUALITY GATE
CTGAN   : REVIEW_REQUIRED - No approved evidence thresholds are defined, so technical completion cannot by itself qualify this output for approval.
TVAE    : REVIEW_REQUIRED - No approved evidence thresholds are defined, so technical completion cannot by itself qualify this output for approval.

REVIEW_REQUIRED means synthetic data was generated, but the available
evidence is not sufficient for unrestricted downstream use. It does
not mean generator training failed; evidence thresholds are not defined
by the project sources, so automatic approval is not justified.
```

The output also states that no generator is approved or declared universally superior. Per-metric comparisons are described as dimension-specific observations, not an overall ranking.

## Warnings

The run emitted SDV warnings recommending metadata JSON saving for future-version reproducibility and noting that the `cuda` parameter is deprecated in a future release. The diagnostic and test runs emitted SDV `FutureWarning` messages that `sdv.load` will be deprecated in favor of `utils.load_synthesizer`. These warnings did not cause failures and were not suppressed. Generator settings and dependency versions were not changed as part of this task.
