"""Deterministic three-class spiral dataset and loaders."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

N_CLASSES = 3
N_PER_CLASS = 500
N_TOTAL = N_CLASSES * N_PER_CLASS
TRAIN_SIZE = 1200
VAL_SIZE = N_TOTAL - TRAIN_SIZE
BATCH_SIZE = 64
NOISE_STD = 0.20
R_MIN = 0.05
R_MAX = 1.0
ANGLE_SPAN = 4.0 * math.pi
CLASS_OFFSET = 2.0 * math.pi / 3.0


def generate_spiral_dataset(
    n_per_class: int = N_PER_CLASS,
    noise_std: float = NOISE_STD,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return features `[3 * n_per_class, 2]` and integer labels `[3 * n_per_class]`."""
    torch.manual_seed(seed)
    radius = torch.linspace(R_MIN, R_MAX, n_per_class)
    base_angle = torch.linspace(0.0, ANGLE_SPAN, n_per_class)

    features = []
    labels = []
    for class_index in range(N_CLASSES):
        noise = torch.randn(n_per_class) * noise_std
        theta = base_angle + class_index * CLASS_OFFSET + noise
        xy = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=1)
        features.append(xy)
        labels.append(torch.full((n_per_class,), class_index, dtype=torch.long))

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def shuffle_and_split(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_size: int = TRAIN_SIZE,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle once, then return an exact train/validation split."""
    torch.manual_seed(seed)
    permutation = torch.randperm(features.shape[0])
    shuffled_x = features[permutation]
    shuffled_y = labels[permutation]
    return (
        shuffled_x[:train_size],
        shuffled_y[:train_size],
        shuffled_x[train_size:],
        shuffled_y[train_size:],
    )


def make_train_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> DataLoader:
    """Training DataLoader with deterministic shuffling."""
    dataset = TensorDataset(features, labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def build_datasets(
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataLoader]:
    """Generate data, split 1,200/300, and wrap the train split in a DataLoader."""
    features, labels = generate_spiral_dataset(seed=seed)
    train_x, train_y, val_x, val_y = shuffle_and_split(features, labels, seed=seed)
    train_loader = make_train_loader(train_x, train_y, seed=seed)
    return train_x, train_y, val_x, val_y, train_loader
