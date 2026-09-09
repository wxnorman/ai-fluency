import json
from pathlib import Path

import torch

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)
    perm = torch.randperm(X.shape[0])
    split_idx = int((1 - test_size) * X.shape[0])
    return X[perm[:split_idx]], X[perm[split_idx:]], y[perm[:split_idx]], y[perm[split_idx:]]

def make_data():
    torch.manual_seed(7)
    X = torch.randn(512, 8)
    w_true = torch.linspace(-1.0, 1.0, 8)
    b_true = torch.tensor(-0.4)
    y = X @ w_true + b_true + 0.05 * torch.randn(512)
    return X, y, w_true, b_true

def mse(pred, target):
    return ((pred - target) ** 2).mean()

def train_model(train_X, train_y, val_X, val_y, w_true, b_true, learning_rate=0.01, num_epochs=1000):
    w = torch.randn_like(w_true, requires_grad=True)
    b = torch.randn_like(b_true, requires_grad=True)
    for epoch in range(num_epochs):
        pred = train_X @ w + b
        loss = mse(pred, train_y)
        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            w.grad.zero_()
            b.grad.zero_()
        if epoch % 25 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return w, b

def evaluate(w, b, val_X, val_y, w_true, b_true):
    with torch.no_grad():
        return {
            "val_mse": mse(val_X @ w + b, val_y).item(),
            "true_mse": mse(val_X @ w_true + b_true, val_y).item(),
            "w_mae": (w - w_true).abs().mean().item(),
            "b_abs_diff": (b - b_true).abs().item(),
        }

def write_metrics(metrics, path="metrics.json"):
    Path(path).write_text(json.dumps(metrics, indent=2) + "\n")

def main() -> None:
    torch.manual_seed(7)
    X, y, w_true, b_true = make_data()
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=7)
    w, b = train_model(train_X, train_y, val_X, val_y, w_true, b_true)
    metrics = evaluate(w, b, val_X, val_y, w_true, b_true)
    write_metrics(metrics)
    print(f"w: {w}")
    print(f"b: {b}")
    print(f"w_true: {w_true}")
    print(f"b_true: {b_true}")
    print(f"mse: {metrics['val_mse']}")
    print(f"mse_true: {metrics['true_mse']}")
    print(f"w_diff: {w - w_true}")
    print(f"b_diff: {b - b_true}")

if __name__ == "__main__":
    main()
