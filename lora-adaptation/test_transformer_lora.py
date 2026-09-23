import torch

from model import TinyCausalTransformer
from transformer_lora import inject_qkv_lora


def test_injection_preserves_initial_logits() -> None:
    torch.manual_seed(0)

    model = TinyCausalTransformer(dropout=0.0)
    model.eval()

    token_ids = torch.randint(0, 13, (4, 8))

    with torch.inference_mode():
        expected = model(token_ids)

    inject_qkv_lora(model, rank=4, alpha=8)
    model.eval()

    with torch.inference_mode():
        actual = model(token_ids)

    torch.testing.assert_close(actual, expected)


def test_only_qkv_lora_parameters_are_trainable() -> None:
    model = TinyCausalTransformer(dropout=0.0)
    inject_qkv_lora(model, rank=4, alpha=8)

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert all(
        name.endswith(("lora_a", "lora_b"))
        for name in trainable
    )

    assert len(trainable) == 4

    trainable_count = sum(
        parameter.numel() for parameter in trainable.values()
    )
    assert trainable_count == 2048
