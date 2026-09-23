import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank)
        )

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)

        if not self.enabled:
            return base_output

        lora_output = F.linear(
            F.linear(x, self.lora_a),
            self.lora_b,
        )

        return base_output + self.scaling * lora_output

    def to_merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            in_features=self.base.in_features,
            out_features=self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )

        with torch.no_grad():
            merged.weight.copy_(
                self.base.weight
                + self.scaling * (self.lora_b @ self.lora_a)
            )

            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)

        return merged
