"""Train the spiral MLP with early stopping and write artifacts."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from data import TRAIN_SIZE, VAL_SIZE, build_datasets
from evaluate import evaluate, load_model_from_checkpoint
from model import DEFAULT_ARCHITECTURE, build_model, parameter_count

SEED = 42
MAX_EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoint.pt"
METRICS_PATH = ROOT / "metrics.json"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def fit(
    model: nn.Module,
    train_loader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    seed: int = SEED,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_x, val_y, criterion=criterion)
        val_loss = float(val_metrics["loss"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "seed": seed,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "architecture": model.architecture(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def metrics_payload(
    seed: int,
    best_epoch: int,
    train_metrics: dict[str, torch.Tensor],
    val_metrics: dict[str, torch.Tensor],
    n_parameters: int,
    elapsed_seconds: float,
) -> dict:
    return {
        "seed": seed,
        "dataset_sizes": {
            "train": TRAIN_SIZE,
            "validation": VAL_SIZE,
            "total": TRAIN_SIZE + VAL_SIZE,
        },
        "best_epoch": best_epoch,
        "train_loss": float(train_metrics["loss"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "confusion_matrix": val_metrics["confusion_matrix"].tolist(),
        "per_class_metrics": {
            "precision": val_metrics["per_class_precision"].tolist(),
            "recall": val_metrics["per_class_recall"].tolist(),
        },
        "macro_metrics": {
            "precision": float(val_metrics["macro_precision"]),
            "recall": float(val_metrics["macro_recall"]),
        },
        "parameter_count": n_parameters,
        "elapsed_seconds": elapsed_seconds,
    }


def write_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def verify_checkpoint_logits(
    model: nn.Module,
    checkpoint_path: Path,
    val_x: torch.Tensor,
    batch_size: int = 64,
) -> None:
    fixed_batch = val_x[:batch_size]
    model.eval()
    with torch.no_grad():
        original_logits = model(fixed_batch)

    reloaded, _ = load_model_from_checkpoint(checkpoint_path)
    with torch.no_grad():
        reloaded_logits = reloaded(fixed_batch)

    if not torch.allclose(original_logits, reloaded_logits):
        raise RuntimeError("Reloaded checkpoint logits do not match the best model.")


def run_pipeline(
    output_dir: Path | None = None,
    seed: int = SEED,
) -> dict:
    output_dir = Path(output_dir) if output_dir is not None else ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"

    set_seed(seed)
    train_x, train_y, val_x, val_y, train_loader = build_datasets(seed=seed)
    model = build_model(**DEFAULT_ARCHITECTURE).to(DEVICE)

    started = time.perf_counter()
    result = fit(model, train_loader, val_x, val_y, seed=seed)
    elapsed = time.perf_counter() - started

    model = result["model"]
    save_checkpoint(
        checkpoint_path,
        model,
        result["optimizer"],
        result["best_epoch"],
        seed,
        result["best_val_loss"],
    )
    verify_checkpoint_logits(model, checkpoint_path, val_x)

    criterion = result["criterion"]
    train_metrics = evaluate(model, train_x, train_y, criterion=criterion)
    val_metrics = evaluate(model, val_x, val_y, criterion=criterion)
    payload = metrics_payload(
        seed=seed,
        best_epoch=result["best_epoch"],
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        n_parameters=parameter_count(model),
        elapsed_seconds=elapsed,
    )
    write_metrics(payload, metrics_path)
    return payload


def main() -> None:
    metrics = run_pipeline()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
