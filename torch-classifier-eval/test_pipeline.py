"""Required tests for the spiral classifier pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

from data import (
    N_PER_CLASS,
    TRAIN_SIZE,
    VAL_SIZE,
    generate_spiral_dataset,
    shuffle_and_split,
)
from evaluate import confusion_matrix, evaluate, precision_recall_from_cm
from model import DEFAULT_ARCHITECTURE, build_model
from train import CHECKPOINT_PATH, METRICS_PATH, SEED, run_pipeline, set_seed

ROOT = Path(__file__).resolve().parent


def _comparable_metrics(metrics: dict) -> dict:
    clone = json.loads(json.dumps(metrics))
    clone.pop("elapsed_seconds", None)
    return clone


def test_python_version():
    assert sys.version_info >= (3, 10)


def test_dataset_shapes_dtypes_counts_and_split():
    features, labels = generate_spiral_dataset(seed=SEED)
    assert features.shape == (1500, 2)
    assert labels.shape == (1500,)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert torch.equal(torch.bincount(labels), torch.tensor([N_PER_CLASS] * 3))

    train_x, train_y, val_x, val_y = shuffle_and_split(features, labels, seed=SEED)
    assert train_x.shape == (TRAIN_SIZE, 2)
    assert train_y.shape == (TRAIN_SIZE,)
    assert val_x.shape == (VAL_SIZE, 2)
    assert val_y.shape == (VAL_SIZE,)
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.int64


def test_dataset_and_split_are_deterministic():
    x1, y1 = generate_spiral_dataset(seed=SEED)
    x2, y2 = generate_spiral_dataset(seed=SEED)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)

    split1 = shuffle_and_split(x1, y1, seed=SEED)
    split2 = shuffle_and_split(x2, y2, seed=SEED)
    for left, right in zip(split1, split2):
        assert torch.equal(left, right)


def test_forward_shape_is_logits():
    set_seed(SEED)
    model = build_model(**DEFAULT_ARCHITECTURE)
    model.eval()
    batch = torch.randn(16, 2)
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (16, 3)
    probabilities = torch.softmax(logits, dim=1)
    assert not torch.allclose(logits, probabilities)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(16), atol=1e-6)
    assert (logits < 0).any()


def test_dropout_train_differs_eval_matches():
    set_seed(SEED)
    model = build_model(**DEFAULT_ARCHITECTURE)
    features = torch.randn(32, 2)

    model.train()
    train_a = model(features)
    train_b = model(features)
    assert not torch.equal(train_a, train_b)

    model.eval()
    with torch.no_grad():
        eval_a = model(features)
        eval_b = model(features)
    assert torch.equal(eval_a, eval_b)


def test_confusion_matrix_precision_recall_hand_example():
    y_true = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = torch.tensor([0, 1, 0, 1, 1, 2, 0, 1])
    expected_cm = torch.tensor(
        [
            [2, 1, 0],
            [0, 2, 0],
            [1, 1, 1],
        ]
    )
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert torch.equal(cm, expected_cm)

    precision, recall = precision_recall_from_cm(cm)
    expected_precision = torch.tensor([2 / 3, 0.5, 1.0])
    expected_recall = torch.tensor([2 / 3, 1.0, 1 / 3])
    assert torch.allclose(precision, expected_precision)
    assert torch.allclose(recall, expected_recall)
    assert torch.isclose(precision.mean(), expected_precision.mean())
    assert torch.isclose(recall.mean(), expected_recall.mean())

    zero_true = torch.tensor([0, 0, 1])
    zero_pred = torch.tensor([0, 0, 0])
    zero_cm = confusion_matrix(zero_true, zero_pred, num_classes=3)
    assert torch.equal(
        zero_cm,
        torch.tensor(
            [
                [2, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    zero_precision, zero_recall = precision_recall_from_cm(zero_cm)
    assert torch.allclose(zero_precision, torch.tensor([2 / 3, 0.0, 0.0]))
    assert torch.allclose(zero_recall, torch.tensor([1.0, 0.0, 0.0]))


def test_checkpoint_reload_matches_logits():
    features, labels = generate_spiral_dataset(seed=SEED)
    _, _, val_x, _ = shuffle_and_split(features, labels, seed=SEED)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    original = build_model(**checkpoint["architecture"])
    original.load_state_dict(checkpoint["model_state"])
    original.eval()

    reloaded = build_model(**checkpoint["architecture"])
    reloaded.load_state_dict(checkpoint["model_state"])
    reloaded.eval()

    batch = val_x[:64]
    with torch.no_grad():
        assert torch.allclose(original(batch), reloaded(batch))


def test_quality_gates():
    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["val_accuracy"] >= 0.90
    assert all(value >= 0.85 for value in metrics["per_class_metrics"]["recall"])
    gap = metrics["train_accuracy"] - metrics["val_accuracy"]
    assert gap <= 0.08

    features, labels = generate_spiral_dataset(seed=metrics["seed"])
    train_x, train_y, val_x, val_y = shuffle_and_split(
        features, labels, seed=metrics["seed"]
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = build_model(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate(model, val_x, val_y, criterion=nn.CrossEntropyLoss())
    assert float(result["accuracy"]) >= 0.90
    assert all(float(value) >= 0.85 for value in result["per_class_recall"])
    train_result = evaluate(model, train_x, train_y, criterion=nn.CrossEntropyLoss())
    assert float(train_result["accuracy"]) - float(result["accuracy"]) <= 0.08


def test_two_clean_runs_match_except_elapsed(tmp_path):
    first = run_pipeline(output_dir=tmp_path / "run1", seed=SEED)
    second = run_pipeline(output_dir=tmp_path / "run2", seed=SEED)
    assert _comparable_metrics(first) == _comparable_metrics(second)
