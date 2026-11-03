# Phase 3.4 — Final Test Evaluation of the Real-Only DL Baseline

## 1. Objective

This phase evaluates the frozen `REAL_ONLY_BASELINE` DLNN on the authoritative held-out test partition. No training, model selection, threshold search, preprocessing fit, calibration, or SHAP analysis was performed.

## 2. Data

The Phase 3.1 split contains 378 training rows, 81 validation rows, and 82 test rows. Only `DEEP_LEARNING/data/processed/X_test.npy` and `y_test.npy` were loaded for the final test evaluation. The measured test distribution was 55 negatives (67.07317073170732%) and 27 positives (32.926829268292686%).

## 3. Model

The frozen Phase 3.2 architecture has 58 inputs, hidden dimensions 128, 64, and 32, and one output logit. It contains 18,305 trainable parameters. Sigmoid converts the logits to probabilities during inference.

## 4. Checkpoint

`DEEP_LEARNING/models/dl_baseline_best.pt` was loaded without modification. It is the Phase 3.3 checkpoint selected at epoch 7 by the lowest validation loss (0.5057232975959778). Its epoch and validation loss were checked against the Phase 3.3 training summary, and the state dictionary loaded strictly into the expected architecture.

The model was placed in evaluation mode. Dropout was disabled, BatchNorm used evaluation statistics, and inference ran under `torch.inference_mode()`.

## 5. Evaluation protocol

- The test partition had not been used for training, early stopping, checkpoint selection, or hyperparameter decisions.
- No retraining or weight modification occurred.
- Primary classification threshold: **0.50, predefined and not tuned on test labels**.
- Accuracy, precision, recall, F1, specificity, and sensitivity use labels derived at that fixed threshold.
- ROC-AUC and PR-AUC use predicted probabilities.
- Repeated inference produced identical probabilities.

## 6. Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8902439024390244 |
| Precision | 0.78125 |
| Recall | 0.9259259259259259 |
| F1 | 0.847457627118644 |
| Specificity | 0.8727272727272727 |
| Sensitivity | 0.9259259259259259 |
| ROC-AUC | 0.971043771043771 |
| PR-AUC / Average Precision | 0.9494435072293363 |

| Confusion matrix cell | Count |
|---|---:|
| TN | 48 |
| FP | 7 |
| FN | 2 |
| TP | 25 |

Machine-readable values and per-row predictions are in `DEEP_LEARNING/reports/test_evaluation.json`, `test_evaluation.csv`, `test_predictions.csv`, and `confusion_matrix.csv`.

## 7. Interpretation

The results represent performance on the held-out test partition of the available dataset. They describe this frozen model’s predictions on these 82 records only. They do not establish clinical validity, medical deployment readiness, or superiority over other models.

## 8. Limitations

- The dataset is small, with one held-out test set of 82 records.
- No external validation was performed.
- No clinical validation was performed.
- This baseline uses no synthetic augmentation.
- Phase 1 documented incomplete dataset provenance, clinical context, measurement units, category meanings, and label-generation information.
- The test evaluation does not establish performance in other hospitals or populations.

## 9. Integrity and restrictions

SHA-256 hashes of the checkpoint, test arrays, preprocessing pipeline, feature inventory, feature-name metadata, and Phase 3.1 preprocessing report matched their pre-evaluation values. Test arrays were not modified. The evaluation code did not create an optimizer or run any training operation. The full unit suite separately creates an Adam object only to test checkpoint serialization; it performs no optimizer step. No CTGAN/TVAE files, threshold selection, hyperparameter search, calibration, or SHAP computation was used.

The complete Deep Learning test suite passed: 54 tests. No warnings or errors were observed during evaluation.

## 10. Phase status

`PHASE 3.4 COMPLETE`
