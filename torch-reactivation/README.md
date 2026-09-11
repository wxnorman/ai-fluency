# torch-reactivation

Linear regression from scratch with PyTorch: synthetic data, a manual gradient-descent loop, and validation MSE.

## Setup

Use the repo-root virtualenv (Python 3.12):

```bash
# from the ai-fluency repo root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
# from this directory, after activating the root venv
python train.py
```

This fits `y ≈ Xw + b` on 512 examples with 8 features, then writes `metrics.json`.

## Test

```bash
pytest test_train.py
```

## Metrics

`metrics.json` is overwritten each training run with:

- `val_mse`: validation error of the learned weights
- `true_mse`: validation error of the true weights (noise floor)
- `w_mae`: mean absolute difference between learned and true `w`
- `b_abs_diff`: absolute difference between learned and true `b`
