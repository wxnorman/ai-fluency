import pytest
import torch

from data import (
    BOS,
    EOS,
    SEP,
    TRAIN_SIZE,
    VAL_SIZE,
    build_all_sequences,
    make_examples,
    split_sequences,
)
from model import CausalSelfAttention, DecoderBlock, FeedForward


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
