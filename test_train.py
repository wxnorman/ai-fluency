import json
from pathlib import Path

import torch

from train import (
    evaluate,
    make_data,
    mse,
    train_model,
    train_test_split,
    write_metrics,
)


def test_make_data_shapes():
    X, y, w_true, b_true = make_data()
    assert X.shape == (512, 8)
    assert y.shape == (512,)
    assert w_true.shape == (8,)
    assert b_true.ndim == 0


def test_make_data_is_reproducible():
    X1, y1, w1, b1 = make_data()
    X2, y2, w2, b2 = make_data()
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)
    assert torch.equal(w1, w2)
    assert torch.equal(b1, b2)


def test_train_test_split_sizes():
    X, y, _, _ = make_data()
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=7)
    assert train_X.shape[0] == 409
    assert val_X.shape[0] == 103
    assert train_y.shape[0] == 409
    assert val_y.shape[0] == 103


def test_train_test_split_is_reproducible():
    X, y, _, _ = make_data()
    a = train_test_split(X, y, test_size=0.2, random_state=7)
    b = train_test_split(X, y, test_size=0.2, random_state=7)
    for left, right in zip(a, b):
        assert torch.equal(left, right)


def test_mse_is_zero_when_equal():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert mse(t, t).item() == 0.0


def test_train_model_reduces_loss():
    X, y, w_true, b_true = make_data()
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=7)
    torch.manual_seed(0)
    w_short, b_short = train_model(
        train_X, train_y, val_X, val_y, w_true, b_true, num_epochs=1
    )
    torch.manual_seed(0)
    w_long, b_long = train_model(
        train_X, train_y, val_X, val_y, w_true, b_true, num_epochs=150
    )
    with torch.no_grad():
        start = mse(train_X @ w_short + b_short, train_y).item()
        end = mse(train_X @ w_long + b_long, train_y).item()
    assert end < start


def test_write_metrics_json(tmp_path):
    path = tmp_path / "metrics.json"
    metrics = evaluate(
        torch.zeros(8),
        torch.tensor(0.0),
        torch.zeros(4, 8),
        torch.zeros(4),
        torch.zeros(8),
        torch.tensor(0.0),
    )
    write_metrics(metrics, path)
    loaded = json.loads(Path(path).read_text())
    assert set(loaded) == {"val_mse", "true_mse", "w_mae", "b_abs_diff"}
    assert loaded["val_mse"] == 0.0
