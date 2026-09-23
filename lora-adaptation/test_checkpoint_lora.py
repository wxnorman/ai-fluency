from pathlib import Path

import torch

from generate import greedy_generate
from model import TinyCausalTransformer
from transformer_lora import inject_qkv_lora


def load_trained_model() -> TinyCausalTransformer:
    checkpoint = torch.load(
        Path("checkpoint.pt"),
        map_location="cpu",
        weights_only=True,
    )

    model = TinyCausalTransformer(
        **checkpoint["model_configuration"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


def test_checkpoint_reverses_examples() -> None:
    model = load_trained_model()

    cases = [
        ([10, 1, 2, 3, 11], [3, 2, 1, 12]),
        ([10, 4, 5, 6, 11], [6, 5, 4, 12]),
        ([10, 9, 0, 7, 11], [7, 0, 9, 12]),
    ]

    for prompt, expected_completion in cases:
        generated = greedy_generate(
            model,
            torch.tensor(prompt),
            max_new_tokens=4,
        )

        assert generated[5:].tolist() == expected_completion


def test_checkpoint_logits_unchanged_after_injection() -> None:
    model = load_trained_model()

    fixed_inputs = torch.tensor(
        [
            [10, 1, 2, 3, 11, 3, 2, 1],
            [10, 4, 5, 6, 11, 6, 5, 4],
        ]
    )

    with torch.inference_mode():
        before = model(fixed_inputs)

    inject_qkv_lora(model, rank=4, alpha=8)
    model.eval()

    with torch.inference_mode():
        after = model(fixed_inputs)

    torch.testing.assert_close(after, before)
    assert torch.equal(after, before)
