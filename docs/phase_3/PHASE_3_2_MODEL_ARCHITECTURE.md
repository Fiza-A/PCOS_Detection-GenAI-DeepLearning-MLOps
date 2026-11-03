# Phase 3.2 — DLNN Model Architecture

## Architecture

The PyTorch module implements the proposed feed-forward architecture exactly:

```text
Input 58
↓
Dense 128 + ReLU
↓
BatchNorm1d(128)
↓
Dropout 0.30
↓
Dense 64 + ReLU
↓
BatchNorm1d(64)
↓
Dropout 0.20
↓
Dense 32 + ReLU
↓
Dense 1
```

There are no additional hidden layers, convolution/recurrent/attention/residual blocks, batch-normalization layers, or dropout layers. The model configuration is centralized in `ModelConfig` in `DEEP_LEARNING/src/model.py`:

| Setting | Value |
|---|---:|
| `input_dim` | 58 |
| `hidden_1` | 128 |
| `hidden_2` | 64 |
| `hidden_3` | 32 |
| `dropout_1` | 0.30 |
| `dropout_2` | 0.20 |
| `output_dim` | 1 |

Layer weight tensor dimensions are `(128, 58)`, `(64, 128)`, `(32, 64)`, and `(1, 32)`. Including biases and the trainable BatchNorm parameters, the architecture has **18,305 trainable parameters**.

## Output and loss boundary

The final `Linear(32, 1)` returns one logit per input row; there is no sigmoid layer inside the model.

> The network outputs a logit during training. Sigmoid is applied when converting the logit to a PCOS probability. BCEWithLogitsLoss will be used during training for numerical stability.

The conceptual output remains `Dense 1 → Sigmoid → PCOS probability`. `logits_to_probabilities` provides the sigmoid conversion utility. Architecture checks confirmed `(batch, 1)` logits, bounded sigmoid probabilities, and compatibility with `torch.nn.BCEWithLogitsLoss` using a small test tensor only.

## Initialization and reproducibility

`initialize_model(seed=42)` seeds PyTorch before constructing the module. CPU initialization was verified by instantiating two models with the same seed and checking exact equality of every parameter tensor. CUDA is not required.

## Training design reserved for a later phase

No training code or optimizer step was added. The later training phase is expected to use:

- Binary cross-entropy with logits (`BCEWithLogitsLoss`).
- Adam optimizer.
- Class weighting based on the observed training target distribution, if appropriate.
- Early stopping based on validation monitoring.

No learning rate, batch size, epoch count, patience, or other training hyperparameters are specified here.

## Data and synthetic-data restriction

The initial experiment remains `REAL_ONLY_BASELINE`. This architecture phase does not load patient feature arrays, CTGAN/TVAE data, or any other synthetic PCOS records. The only synthetic tensor is a four-row, 58-column random tensor used for shape and loss-compatibility tests; it is not a patient dataset. No prediction metrics or SHAP explanations were calculated.

## Verification

The architecture suite checks model construction, input and hidden dimensions, dropout rates, one-logit output, finite forward output, BCE-with-logits compatibility, sigmoid probability bounds, reproducible initialization, parameter count, and exact module structure. The complete command `.venv\Scripts\python.exe -m unittest discover -s DEEP_LEARNING\tests -v` reran the Phase 3.1 tests and passed all **27 tests**. No model training was performed.

## Status

`PHASE 3.2 COMPLETE`
