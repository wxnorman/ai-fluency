import torch

from data import EOS


def greedy_generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int = 4,
    eos_token: int = EOS,
) -> torch.Tensor:
    if prompt.ndim != 1:
        raise ValueError(
            f"Expected one prompt with shape [T], got {prompt.shape}"
        )

    model.eval()

    device = next(model.parameters()).device
    generated = prompt.to(device).clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.shape[0] > model.context_length:
                raise ValueError(
                    "Generated sequence exceeds model context length"
                )

            logits = model(
                generated.unsqueeze(0)
            )

            next_token = logits[
                0,
                -1,
            ].argmax(dim=-1)

            generated = torch.cat(
                [
                    generated,
                    next_token.view(1),
                ]
            )

            if next_token.item() == eos_token:
                break

    return generated.cpu()


def evaluate_generation(
    model: torch.nn.Module,
    sequences: torch.Tensor,
) -> dict[str, float]:
    token_correct = 0
    token_total = 0
    sequence_correct = 0

    for sequence in sequences:
        prompt = sequence[:5]
        expected_completion = sequence[5:]

        generated = greedy_generate(
            model,
            prompt,
            max_new_tokens=4,
        )

        actual_completion = generated[5:]

        # Missing tokens are counted as incorrect.
        aligned_prediction = torch.full(
            expected_completion.shape,
            fill_value=-1,
            dtype=torch.long,
        )

        copied_length = min(
            actual_completion.shape[0],
            expected_completion.shape[0],
        )

        aligned_prediction[:copied_length] = (
            actual_completion[:copied_length]
        )

        matches = (
            aligned_prediction
            == expected_completion
        )

        token_correct += matches.sum().item()
        token_total += expected_completion.numel()

        sequence_correct += int(matches.all().item())

    return {
        "token_accuracy": token_correct / token_total,
        "exact_sequence_accuracy": (
            sequence_correct / sequences.shape[0]
        ),
        "sequence_count": sequences.shape[0],
    }