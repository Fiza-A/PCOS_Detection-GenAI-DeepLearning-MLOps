# Phase 3.3 — Controlled Real-Only Deep Learning Training

## Experiment

`REAL_ONLY_BASELINE`

The run used the existing Phase 3.1 arrays. It loaded only X/y for training and validation. The test partition was identified from preprocessing metadata as 82 rows, but test arrays were not loaded.

## Dataset

| Partition | Inputs | Labels |
|---|---:|---:|
| Train | 378 × 58 | 378 |
| Validation | 81 × 58 | 81 |
| Test | 82 × 58 (held out; metadata only) | Not loaded |

No new split was created. No CTGAN/TVAE or other synthetic data was loaded.

## Model

The existing Phase 3.2 architecture was used without modification: `58 → 128 → 64 → 32 → 1`, with 18,305 trainable parameters. The network returns logits.

## Training configuration

- Loss: `BCEWithLogitsLoss` with `pos_weight`.
- Positive class weight: calculated only from `y_train` as negatives / positives = `254 / 124 = 2.0483870967741935`.
- Optimizer: Adam.
- Learning rate: 0.001.
- Batch size: 32; batches are constructed from training rows only.
- Maximum epochs: 100.
- Validation: after every epoch; monitored validation loss.
- Early stopping patience: 15 consecutive epochs without validation-loss improvement.
- Seed: 42 for Python, NumPy, PyTorch, and the shuffled training DataLoader.
- Device: CPU; deterministic PyTorch algorithms enabled.

The best checkpoint is saved on each validation-loss improvement. The official run saved the best checkpoint at epoch 7. Checkpoint load verification confirmed its epoch and validation loss match the minimum recorded in the training history.

## Test isolation

**The test partition was not used for training, early stopping, checkpoint selection, or hyperparameter decisions.** The training loader and training function receive training arrays only; validation is used exclusively for validation and early stopping. Test arrays were not opened, and no test metrics were calculated.

## Actual training results

| Result | Observed value |
|---|---:|
| Epochs run | 22 |
| Final epoch | 22 |
| Best epoch | 7 |
| Best validation loss | 0.5057232975959778 |
| Training duration | 7.792364600001747 seconds |

Early stopping occurred after 15 consecutive epochs without improvement following the best epoch. These are training/validation observations only and do not establish final test performance or clinical utility.

## Artifacts

- Best checkpoint: `DEEP_LEARNING/models/dl_baseline_best.pt`
- Epoch history: `DEEP_LEARNING/reports/training_history.csv`
- Training configuration and class-weight provenance: `DEEP_LEARNING/reports/training_config.json`
- Run summary: `DEEP_LEARNING/reports/training_summary.json`
- Training/validation loss curve: `DEEP_LEARNING/reports/training_loss_curve.png`

## Tests

Command: `.venv\Scripts\python.exe -m unittest discover -s DEEP_LEARNING\tests -v`

Result: **37 tests passed**, covering the existing Phase 3.1 preprocessing tests, Phase 3.2 architecture checks, and Phase 3.3 training utilities. The tests do not invoke the official training loop or overwrite its checkpoint.

No warnings or training errors were observed. No test performance metrics, calibration, or SHAP results were produced.

## Status

`PHASE 3.3 COMPLETE`
