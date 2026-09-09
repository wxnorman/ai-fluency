# torch-reactivation

Linear regression from scratch with PyTorch: synthetic data, a manual gradient-descent loop, and validation MSE.

## Setup

```bash
source .venv/bin/activate
```

PyTorch and pytest are already installed in `.venv`.

## Train

```bash
.venv/bin/python train.py
```

This fits `y ≈ Xw + b` on 512 examples with 8 features, then writes `metrics.json`.

## Test

```bash
.venv/bin/pytest test_train.py
```

## Metrics

`metrics.json` is overwritten each training run with:

- `val_mse`: validation error of the learned weights
- `true_mse`: validation error of the true weights (noise floor)
- `w_mae`: mean absolute difference between learned and true `w`
- `b_abs_diff`: absolute difference between learned and true `b`
