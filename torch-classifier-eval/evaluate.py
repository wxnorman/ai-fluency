"""Evaluation metrics implemented with tensor operations only."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from model import MLP


def confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    """Rows are true classes; columns are predicted classes."""
    y_true = y_true.long().view(-1)
    y_pred = y_pred.long().view(-1)
    indices = y_true * num_classes + y_pred
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def precision_recall_from_cm(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class precision and recall. A zero denominator yields 0."""
    cm = cm.float()
    true_positives = torch.diag(cm)
    predicted_positives = cm.sum(dim=0)
    actual_positives = cm.sum(dim=1)
    precision = torch.where(
        predicted_positives > 0,
        true_positives / predicted_positives,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        actual_positives > 0,
        true_positives / actual_positives,
        torch.zeros_like(true_positives),
    )
    return precision, recall


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module | None = None,
    num_classes: int = 3,
) -> dict[str, torch.Tensor]:
    """Run `model.eval()` under `torch.no_grad()` and compute all required metrics."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        logits = model(features)
        mean_loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        cm = confusion_matrix(labels, predictions, num_classes=num_classes)
        precision, recall = precision_recall_from_cm(cm)

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "logits": logits,
        "predictions": predictions,
    }


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> tuple[MLP, dict]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    architecture = dict(checkpoint["architecture"])
    model = MLP(**architecture)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
