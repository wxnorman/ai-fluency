# torch-classifier-eval

Deterministic three-class spiral classification with a small MLP, early stopping, and metrics implemented in PyTorch only.

## Setup

Python 3.10+ is required. This project was run with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch numpy pytest
```

## Train and test

```bash
.venv/bin/python train.py
.venv/bin/pytest -q
```

`train.py` writes `checkpoint.pt` and `metrics.json`. The checkpoint stores model state, optimizer state, the best epoch, seed, architecture parameters, and best validation loss. Reloading that file into a newly constructed model must reproduce the original logits on a fixed validation batch.

## Why `CrossEntropyLoss` receives logits

`torch.nn.CrossEntropyLoss` applies `log_softmax` internally, then the negative log-likelihood of the true class. Feeding softmax probabilities would apply that log-softmax a second time, squash already-normalized scores, and train the wrong objective. The network therefore ends with a linear layer and returns raw logits.

## What `model.train()` and `model.eval()` change here

The only stochastic layer is `Dropout(p=0.10)` before the output. In train mode, dropout zeros hidden units at random, so two forward passes on the same batch differ. In eval mode, dropout is disabled and the hidden activations are used in full, so two forward passes match. BatchNorm is not used, so eval mode does not change any running statistics.

## Why the best checkpoint can outperform the final epoch

Training continues after the validation loss has already reached its minimum. Later epochs can overfit the 1,200 training points or wander because of dropout noise, so validation loss rises. Early stopping keeps the weights from the lowest validation-loss epoch (patience 15) rather than the last epoch.

## Confusion-matrix orientation and zero denominators

`confusion_matrix[i, j]` is the count of examples whose true class is `i` and whose predicted class is `j` (rows = true, columns = predicted).

Precision for class `k` is `cm[k, k] / sum_i cm[i, k]`. Recall for class `k` is `cm[k, k] / sum_j cm[k, j]`. If a class is never predicted, precision is defined as 0. If a class never appears in the labels, recall is defined as 0. Those zeros are also what enter the macro averages.

## Debugging narrative

**Symptom.** Two supposedly clean runs with the same seed wrote different `metrics.json` values, so the reproducibility test failed.

**Hypothesis.** Either the spiral noise was not seeded, or the training `DataLoader` was shuffling with the global RNG and advancing it differently across runs.

**Diagnostic.** Printing `train_x[:5]` matched across runs, so dataset generation was fine. Printing the first batch of labels from the `DataLoader` did not match when the loader was created with `shuffle=True` and no `generator`.

**Cause.** `DataLoader` shuffle uses an internal generator that is not reset by `torch.manual_seed` alone in a way that is isolated from later RNG use. Without an explicit `torch.Generator().manual_seed(seed)`, batch order (and therefore dropout masks and AdamW updates) diverged.

**Fix.** `make_train_loader` now builds a dedicated `Generator`, seeds it, and passes it to `DataLoader`. Combined with `set_seed` at the start of `run_pipeline`, two clean runs match on every metric except elapsed time.
