"""MLP classifier: 2 → 64 → 64 → 3 with ReLU and dropout before the logits."""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_ARCHITECTURE = {
    "input_dim": 2,
    "hidden_dim": 64,
    "num_classes": 3,
    "dropout": 0.10,
}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def architecture(self) -> dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout_p,
        }


def build_model(
    input_dim: int = 2,
    hidden_dim: int = 64,
    num_classes: int = 3,
    dropout: float = 0.10,
) -> MLP:
    return MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
