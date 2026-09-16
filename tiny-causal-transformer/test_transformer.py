import pytest
import torch

from data import (
    BOS,
    EOS,
    SEP,
    TRAIN_SIZE,
    VAL_SIZE,
    VOCABULARY,
    VOCAB_SIZE,
    build_all_sequences,
    make_data_loader,
    make_examples,
    split_sequences,
)
from generate import evaluate_generation, greedy_generate
from model import CausalSelfAttention, DecoderBlock, FeedForward, TinyCausalTransformer
from train import (
    completion_counts,
    evaluate_teacher_forced,
    gradient_norm_summary,
    load_checkpoint,
    masked_cross_entropy,
    save_checkpoint,
)


def test_full_shape_is_1000_by_9():
    sequences = build_all_sequences()
    assert sequences.shape == (1000, 9)


def test_dtype_is_int64():
    sequences = build_all_sequences()
    assert sequences.dtype == torch.int64


def test_every_token_is_between_0_and_12():
    sequences = build_all_sequences()
    assert sequences.min().item() >= 0
    assert sequences.max().item() <= 12


def test_all_prompt_triples_are_unique():
    sequences = build_all_sequences()
    prompts = sequences[:, 1:4]
    unique = torch.unique(prompts, dim=0)
    assert unique.shape[0] == 1000


def test_columns_5_to_7_reverse_columns_1_to_3():
    sequences = build_all_sequences()
    assert torch.equal(sequences[:, 5:8], sequences[:, 1:4].flip(dims=(1,)))


def test_special_tokens_occupy_columns_0_4_and_8():
    sequences = build_all_sequences()
    assert torch.equal(sequences[:, 0], torch.full((1000,), BOS))
    assert torch.equal(sequences[:, 4], torch.full((1000,), SEP))
    assert torch.equal(sequences[:, 8], torch.full((1000,), EOS))


def test_split_sizes_are_800_and_200():
    train, val = split_sequences(build_all_sequences())
    assert train.shape[0] == TRAIN_SIZE
    assert val.shape[0] == VAL_SIZE
    assert train.shape[1] == 9
    assert val.shape[1] == 9


def test_train_and_val_triples_do_not_overlap():
    train, val = split_sequences(build_all_sequences())
    combined = torch.cat((train[:, 1:4], val[:, 1:4]), dim=0)
    assert torch.unique(combined, dim=0).shape[0] == 1000


def test_same_seed_produces_identical_splits():
    sequences = build_all_sequences()
    train_a, val_a = split_sequences(sequences, seed=42)
    train_b, val_b = split_sequences(sequences, seed=42)
    assert torch.equal(train_a, train_b)
    assert torch.equal(val_a, val_b)


def test_inputs_and_targets_have_shape_n_by_8():
    sequences = build_all_sequences()
    inputs, targets, _ = make_examples(sequences)
    assert inputs.shape == (1000, 8)
    assert targets.shape == (1000, 8)


def test_inputs_shifted_equal_targets_unshifted():
    sequences = build_all_sequences()
    inputs, targets, _ = make_examples(sequences)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_loss_mask_is_bool_with_four_trues_per_row():
    sequences = build_all_sequences()
    _, _, loss_mask = make_examples(sequences)
    assert loss_mask.dtype == torch.bool
    assert loss_mask.shape == (1000, 8)
    assert torch.equal(loss_mask.sum(dim=1), torch.full((1000,), 4))
    expected = torch.tensor([False, False, False, False, True, True, True, True])
    assert torch.equal(loss_mask, expected.expand_as(loss_mask))


def test_d_model_must_be_divisible_by_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(d_model=64, n_heads=5)


def test_causal_mask_is_lower_triangular_bool_buffer():
    attn = CausalSelfAttention()
    mask = attn.causal_mask
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 8, 8)
    expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    assert torch.equal(mask[0, 0], expected)
    assert "causal_mask" in dict(attn.named_buffers())
    assert "causal_mask" not in dict(attn.named_parameters())


def test_project_qkv_shapes():
    attn = CausalSelfAttention()
    x = torch.randn(2, 8, 64)
    q, k, v = attn.project_qkv(x)
    assert q.shape == (2, 4, 8, 16)
    assert k.shape == (2, 4, 8, 16)
    assert v.shape == (2, 4, 8, 16)


def test_split_and_merge_heads_roundtrip():
    attn = CausalSelfAttention()
    x = torch.randn(3, 8, 64)
    merged = attn.merge_heads(attn.split_heads(x))
    assert torch.allclose(merged, x)


def test_attention_scores_shape_and_scale():
    attn = CausalSelfAttention()
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    scores = attn.compute_attention_scores(q, k)
    assert scores.shape == (2, 4, 8, 8)
    expected = (q @ k.transpose(-2, -1)) / (16 ** 0.5)
    assert torch.allclose(scores, expected)


def test_attention_probabilities_are_causal_and_sum_to_one():
    attn = CausalSelfAttention()
    scores = torch.zeros(2, 4, 8, 8)
    probs = attn.compute_attention_probabilities(scores)
    assert probs.shape == (2, 4, 8, 8)
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.equal(probs[:, :, future], torch.zeros(2, 4, 8, 8)[:, :, future])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4, 8))


def test_forward_shape_matches_input():
    attn = CausalSelfAttention()
    x = torch.randn(2, 8, 64)
    output = attn(x)
    assert output.shape == x.shape


def test_forward_return_attention_matches_unmasked_probs():
    attn = CausalSelfAttention()
    attn.eval()
    x = torch.randn(2, 8, 64)
    output, probs = attn(x, return_attention=True)
    assert output.shape == x.shape
    assert probs.shape == (2, 4, 8, 8)
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.equal(probs[:, :, future], torch.zeros_like(probs)[:, :, future])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4, 8))


def test_eval_forward_is_deterministic():
    attn = CausalSelfAttention(dropout=0.1)
    attn.eval()
    x = torch.randn(2, 8, 64)
    first = attn(x)
    second = attn(x)
    assert torch.equal(first, second)


def test_feedforward_layers_are_linear_gelu_linear_dropout():
    ff = FeedForward()
    types = [type(layer) for layer in ff.net]
    assert types == [torch.nn.Linear, torch.nn.GELU, torch.nn.Linear, torch.nn.Dropout]
    assert ff.net[0].in_features == 64
    assert ff.net[0].out_features == 256
    assert ff.net[2].in_features == 256
    assert ff.net[2].out_features == 64


def test_feedforward_preserves_token_shape():
    ff = FeedForward()
    x = torch.randn(2, 8, 64)
    assert ff(x).shape == x.shape


def test_feedforward_eval_is_deterministic_train_dropout_differs():
    ff = FeedForward(dropout=0.1)
    x = torch.randn(2, 8, 64)

    ff.train()
    train_a = ff(x)
    train_b = ff(x)
    assert not torch.equal(train_a, train_b)

    ff.eval()
    eval_a = ff(x)
    eval_b = ff(x)
    assert torch.equal(eval_a, eval_b)


def test_decoder_block_is_prenorm_with_residuals():
    block = DecoderBlock()
    assert isinstance(block.attention_norm, torch.nn.LayerNorm)
    assert isinstance(block.feed_forward_norm, torch.nn.LayerNorm)
    assert block.attention_norm.normalized_shape == (64,)
    assert block.feed_forward_norm.normalized_shape == (64,)
    assert isinstance(block.attention, CausalSelfAttention)
    assert isinstance(block.feed_forward, FeedForward)


def test_decoder_block_preserves_shape():
    block = DecoderBlock()
    x = torch.randn(2, 8, 64)
    assert block(x).shape == x.shape


def test_decoder_block_return_attention_is_causal():
    block = DecoderBlock()
    block.eval()
    x = torch.randn(2, 8, 64)
    output, probs = block(x, return_attention=True)
    assert output.shape == x.shape
    assert probs.shape == (2, 4, 8, 8)
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.equal(probs[:, :, future], torch.zeros_like(probs)[:, :, future])
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4, 8))


def test_decoder_block_eval_is_deterministic_train_dropout_differs():
    block = DecoderBlock(dropout=0.1)
    x = torch.randn(2, 8, 64)

    block.train()
    train_a = block(x)
    train_b = block(x)
    assert not torch.equal(train_a, train_b)

    block.eval()
    eval_a = block(x)
    eval_b = block(x)
    assert torch.equal(eval_a, eval_b)

def test_tiny_transformer_output_shapes():
    torch.manual_seed(42)

    model = TinyCausalTransformer(dropout=0.0)
    token_ids = torch.randint(0, 13, (2, 8))

    logits, attention_maps = model(
        token_ids,
        return_attention=True,
    )

    assert logits.shape == (2, 8, 13)
    assert len(attention_maps) == 2

    for probabilities in attention_maps:
        assert probabilities.shape == (2, 4, 8, 8)

def test_tiny_transformer_is_causal():
    torch.manual_seed(42)

    model = TinyCausalTransformer(dropout=0.0)
    model.eval()

    original = torch.randint(0, 13, (2, 8))
    changed = original.clone()

    changed[:, 5:] = torch.randint(
        0,
        13,
        changed[:, 5:].shape,
    )

    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)

    assert torch.allclose(
        original_logits[:, :5],
        changed_logits[:, :5],
        atol=1e-6,
        rtol=1e-5,
    )


def test_model_configuration_reconstructs_model():
    original = TinyCausalTransformer()

    reconstructed = TinyCausalTransformer(
        **original.configuration()
    )

    assert reconstructed.configuration() == original.configuration()


def test_tiny_transformer_rejects_context_overflow():
    model = TinyCausalTransformer(context_length=8)
    token_ids = torch.randint(0, 13, (2, 9))

    with pytest.raises(ValueError, match="exceeds"):
        model(token_ids)


def test_masked_loss_ignores_prompt_targets():
    torch.manual_seed(42)

    logits = torch.randn(2, 8, 13)

    original_targets = torch.randint(
        0,
        13,
        (2, 8),
    )

    changed_targets = original_targets.clone()

    # Change only prompt-region targets.
    changed_targets[:, :4] = (
        changed_targets[:, :4] + 1
    ) % 13

    loss_mask = torch.tensor(
        [False, False, False, False, True, True, True, True]
    ).expand(2, 8)

    original_loss = masked_cross_entropy(
        logits,
        original_targets,
        loss_mask,
    )

    changed_loss = masked_cross_entropy(
        logits,
        changed_targets,
        loss_mask,
    )

    assert torch.equal(
        original_loss,
        changed_loss,
    )


def test_masked_loss_has_zero_prompt_gradients():
    torch.manual_seed(42)

    logits = torch.randn(
        2,
        8,
        13,
        requires_grad=True,
    )

    targets = torch.randint(0, 13, (2, 8))

    loss_mask = torch.tensor(
        [False, False, False, False, True, True, True, True]
    ).expand(2, 8)

    loss = masked_cross_entropy(
        logits,
        targets,
        loss_mask,
    )

    loss.backward()

    assert logits.grad is not None

    prompt_gradients = logits.grad[:, :4]
    completion_gradients = logits.grad[:, 4:]

    assert torch.count_nonzero(prompt_gradients) == 0
    assert torch.count_nonzero(completion_gradients) > 0
    assert torch.isfinite(logits.grad).all()


def test_masked_loss_reaches_required_model_parameters():
    torch.manual_seed(42)

    sequences = build_all_sequences()
    train_sequences, _ = split_sequences(
        sequences,
        seed=42,
    )

    inputs, targets, loss_mask = make_examples(
        train_sequences[:32]
    )

    model = TinyCausalTransformer(dropout=0.0)

    logits = model(inputs)

    assert logits.shape == (32, 8, 13)

    loss = masked_cross_entropy(
        logits,
        targets,
        loss_mask,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()

    required_parameters = {
        "token_embedding": model.token_embedding.weight,
        "position_embedding": model.position_embedding.weight,
        "first_qkv": (
            model.blocks[0]
            .attention
            .qkv_projection
            .weight
        ),
        "output_projection": model.output_projection.weight,
    }

    for name, parameter in required_parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm().item() > 0, name


def test_gradient_norm_summary():
    torch.manual_seed(42)

    sequences = build_all_sequences()
    train_sequences, _ = split_sequences(sequences)

    inputs, targets, loss_mask = make_examples(
        train_sequences[:32]
    )

    model = TinyCausalTransformer(dropout=0.0)

    loss = masked_cross_entropy(
        model(inputs),
        targets,
        loss_mask,
    )
    loss.backward()

    summary = gradient_norm_summary(model)

    assert "token_embedding.weight" in summary
    assert "position_embedding.weight" in summary
    assert "blocks.0.attention.qkv_projection.weight" in summary
    assert "output_projection.weight" in summary

    assert all(
        value > 0
        for value in summary.values()
    )


class _FixedNextTokenModel(torch.nn.Module):
    """Emits a predetermined token from the last time step each forward pass."""

    def __init__(self, next_tokens: list[int], context_length: int = 8) -> None:
        super().__init__()
        self.context_length = context_length
        self._next_tokens = list(next_tokens)
        self._step = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, time = token_ids.shape
        if time == 5:
            self._step = 0
        logits = torch.zeros(batch_size, time, VOCAB_SIZE)
        token = self._next_tokens[min(self._step, len(self._next_tokens) - 1)]
        self._step += 1
        logits[:, 0, 0] = 50.0
        logits[:, -1, token] = 100.0
        return logits


def test_vocabulary_covers_digits_and_special_tokens():
    assert len(VOCABULARY) == VOCAB_SIZE
    assert VOCABULARY["BOS"] == BOS
    assert VOCABULARY["SEP"] == SEP
    assert VOCABULARY["EOS"] == EOS
    for digit in range(10):
        assert VOCABULARY[str(digit)] == digit


def test_data_loader_yields_masked_batches():
    sequences = build_all_sequences()[:128]
    loader = make_data_loader(sequences, batch_size=64, shuffle=False, seed=42)
    inputs, targets, loss_mask = next(iter(loader))
    assert inputs.shape == (64, 8)
    assert targets.shape == (64, 8)
    assert loss_mask.shape == (64, 8)
    assert loss_mask.dtype == torch.bool
    expected_mask = torch.tensor(
        [False, False, False, False, True, True, True, True]
    )
    assert torch.equal(loss_mask, expected_mask.expand_as(loss_mask))
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_data_loader_shuffle_is_deterministic():
    sequences = build_all_sequences()[:128]
    first = next(iter(make_data_loader(sequences, batch_size=64, shuffle=True, seed=42)))
    second = next(iter(make_data_loader(sequences, batch_size=64, shuffle=True, seed=42)))
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_greedy_generate_rejects_batched_prompt():
    model = TinyCausalTransformer(dropout=0.0)
    with pytest.raises(ValueError, match="one prompt"):
        greedy_generate(model, torch.randint(0, 13, (2, 5)))


def test_greedy_generate_stops_at_eos_and_uses_last_position():
    prompt = torch.tensor([BOS, 1, 2, 3, SEP])
    model = _FixedNextTokenModel([3, 2, 1, EOS])
    generated = greedy_generate(model, prompt, max_new_tokens=4)
    assert generated.tolist() == [BOS, 1, 2, 3, SEP, 3, 2, 1, EOS]


def test_greedy_generate_stops_early_when_eos_appears():
    prompt = torch.tensor([BOS, 1, 2, 3, SEP])
    model = _FixedNextTokenModel([EOS, 9, 9, 9])
    generated = greedy_generate(model, prompt, max_new_tokens=4)
    assert generated.tolist() == [BOS, 1, 2, 3, SEP, EOS]


def test_greedy_generate_rejects_context_overflow():
    model = _FixedNextTokenModel([0], context_length=8)
    prompt = torch.arange(9)
    with pytest.raises(ValueError, match="exceeds"):
        greedy_generate(model, prompt, max_new_tokens=1)


def test_evaluate_generation_metrics_on_known_reverse():
    sequences = torch.tensor(
        [
            [BOS, 1, 2, 3, SEP, 3, 2, 1, EOS],
            [BOS, 4, 5, 6, SEP, 6, 5, 4, EOS],
        ]
    )
    model = _FixedNextTokenModel([3, 2, 1, EOS])
    metrics = evaluate_generation(model, sequences)
    assert metrics["sequence_count"] == 2
    assert metrics["token_accuracy"] == 0.625
    assert metrics["exact_sequence_accuracy"] == 0.5


def test_completion_counts_ignore_prompt_mismatches():
    logits = torch.zeros(2, 8, 13)
    targets = torch.zeros(2, 8, dtype=torch.long)
    targets[:, 4:] = 1
    logits[:, :, 1] = 10.0
    loss_mask = torch.tensor(
        [False, False, False, False, True, True, True, True]
    ).expand(2, 8)

    counts = completion_counts(logits, targets, loss_mask)
    assert counts["token_correct"] == 8
    assert counts["token_total"] == 8
    assert counts["sequence_correct"] == 2
    assert counts["sequence_total"] == 2


def test_evaluate_teacher_forced_sets_eval_mode():
    sequences = build_all_sequences()[:64]
    loader = make_data_loader(sequences, batch_size=32, shuffle=False, seed=42)
    model = TinyCausalTransformer(dropout=0.1)
    model.train()
    metrics = evaluate_teacher_forced(model, loader)
    assert not model.training
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert 0.0 <= metrics["exact_sequence_accuracy"] <= 1.0
    assert metrics["sequence_count"] == 64
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_masked_loss_rejects_empty_mask_and_non_bool():
    logits = torch.randn(1, 8, 13)
    targets = torch.zeros(1, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="no active tokens"):
        masked_cross_entropy(
            logits,
            targets,
            torch.zeros(1, 8, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="boolean"):
        masked_cross_entropy(
            logits,
            targets,
            torch.ones(1, 8),
        )


def test_checkpoint_reload_matches_logits(tmp_path):
    torch.manual_seed(0)
    model = TinyCausalTransformer(dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        seed=42,
        best_epoch=7,
        best_validation_loss=0.25,
    )

    reloaded, checkpoint = load_checkpoint(path)
    assert checkpoint["seed"] == 42
    assert checkpoint["best_epoch"] == 7
    assert checkpoint["vocabulary"] == VOCABULARY
    assert checkpoint["model_configuration"] == model.configuration()

    token_ids = torch.randint(0, 13, (2, 8))
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(model(token_ids), reloaded(token_ids))