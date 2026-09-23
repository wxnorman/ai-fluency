import torch
from torch import nn

from lora import LoRALinear


def test_initial_output_matches_base_layer() -> None:
    torch.manual_seed(0)

    base = nn.Linear(8, 6)
    x = torch.randn(4, 8)

    expected = base(x)
    lora = LoRALinear(base, rank=2, alpha=4)
    actual = lora(x)

    torch.testing.assert_close(actual, expected)


def test_first_backward_gradient_behavior() -> None:
    torch.manual_seed(0)

    base = nn.Linear(8, 6)
    lora = LoRALinear(base, rank=2, alpha=4)

    x = torch.randn(4, 8)
    target = torch.randn(4, 6)

    loss = torch.nn.functional.mse_loss(lora(x), target)
    loss.backward()

    assert base.weight.grad is None
    assert base.bias.grad is None

    assert lora.lora_a.grad is not None
    assert lora.lora_b.grad is not None

    assert torch.count_nonzero(lora.lora_a.grad) == 0
    assert torch.count_nonzero(lora.lora_b.grad) > 0


def test_only_lora_parameters_are_trainable() -> None:
    base = nn.Linear(8, 6)
    lora = LoRALinear(base, rank=2, alpha=4)

    trainable = {
        name: parameter
        for name, parameter in lora.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {"lora_a", "lora_b"}

    trainable_count = sum(
        parameter.numel() for parameter in trainable.values()
    )
    assert trainable_count == 2 * (8 + 6)


def test_a_receives_gradient_after_b_updates() -> None:
    torch.manual_seed(0)

    base = nn.Linear(8, 6)
    original_weight = base.weight.detach().clone()
    original_bias = base.bias.detach().clone()

    lora = LoRALinear(base, rank=2, alpha=4)

    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in lora.parameters()
            if parameter.requires_grad
        ],
        lr=0.1,
    )

    x = torch.randn(4, 8)
    target = torch.randn(4, 6)

    # First backward pass
    first_loss = torch.nn.functional.mse_loss(lora(x), target)
    first_loss.backward()

    first_a_norm = lora.lora_a.grad.norm().item()
    first_b_norm = lora.lora_b.grad.norm().item()

    assert first_a_norm == 0.0
    assert first_b_norm > 0.0

    optimizer.step()

    # B should now be nonzero.
    assert torch.count_nonzero(lora.lora_b) > 0

    optimizer.zero_grad(set_to_none=True)

    # Second backward pass
    second_loss = torch.nn.functional.mse_loss(lora(x), target)
    second_loss.backward()

    second_a_norm = lora.lora_a.grad.norm().item()
    second_b_norm = lora.lora_b.grad.norm().item()

    print(
        f"first gradients:  A={first_a_norm:.6f}, "
        f"B={first_b_norm:.6f}"
    )
    print(
        f"second gradients: A={second_a_norm:.6f}, "
        f"B={second_b_norm:.6f}"
    )

    assert second_a_norm > 0.0
    assert second_b_norm > 0.0

    # The frozen layer must remain unchanged.
    torch.testing.assert_close(base.weight, original_weight)
    torch.testing.assert_close(base.bias, original_bias)


def test_merged_layer_matches_lora_output() -> None:
    torch.manual_seed(0)

    base = nn.Linear(8, 6)
    lora = LoRALinear(base, rank=2, alpha=4)

    with torch.no_grad():
        lora.lora_b.normal_()

    x = torch.randn(5, 8)

    expected = lora(x)
    merged = lora.to_merged_linear()
    actual = merged(x)

    torch.testing.assert_close(actual, expected)
    assert isinstance(merged, nn.Linear)
    assert set(dict(merged.named_parameters())) == {
        "weight",
        "bias",
    }
