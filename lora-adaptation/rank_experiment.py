import copy

import torch
from torch import nn
from torch.nn import functional as F

from lora import LoRALinear


torch.manual_seed(0)

IN_FEATURES = 16
OUT_FEATURES = 12
TRUE_RANK = 4

left, _ = torch.linalg.qr(
    torch.randn(OUT_FEATURES, TRUE_RANK)
)
right, _ = torch.linalg.qr(
    torch.randn(IN_FEATURES, TRUE_RANK)
)

singular_values = torch.tensor([1.0, 0.8, 0.6, 0.4])

target_delta = (
    left
    @ torch.diag(singular_values)
    @ right.T
)

print("target rank:", torch.linalg.matrix_rank(target_delta).item())

base = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
x = torch.eye(IN_FEATURES)

with torch.no_grad():
    target = F.linear(
        x,
        base.weight + target_delta,
    )


def train_adapter(rank: int) -> tuple[float, float]:
    layer = LoRALinear(
        base=copy.deepcopy(base),
        rank=rank,
        alpha=rank,
    )

    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in layer.parameters()
            if parameter.requires_grad
        ],
        lr=0.05,
    )

    for _ in range(2000):
        optimizer.zero_grad(set_to_none=True)

        prediction = layer(x)
        loss = F.mse_loss(prediction, target)

        loss.backward()
        optimizer.step()

    effective_delta = (
        layer.scaling
        * layer.lora_b
        @ layer.lora_a
    )

    relative_error = (
        (effective_delta - target_delta).norm()
        / target_delta.norm()
    ).item()

    return loss.item(), relative_error


for rank in [1, 4]:
    loss, relative_error = train_adapter(rank)

    print(
        f"rank={rank}: "
        f"loss={loss:.8f}, "
        f"relative_weight_error={relative_error:.6f}"
    )
