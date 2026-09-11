import torch

BOS = 10
SEP = 11
EOS = 12
VOCAB_SIZE = 13

SEQUENCE_LENGTH = 9
CONTEXT_LENGTH = 8
TRAIN_SIZE = 800
VAL_SIZE = 200


def build_all_sequences() -> torch.Tensor:
    """Return all 1,000 sequences with shape [1000, 9].

    Each row is ``[BOS, a, b, c, SEP, c, b, a, EOS]`` for ``a, b, c`` in
    ``{0, ..., 9}``.
    """
    digits = torch.arange(10)
    triples = torch.cartesian_prod(digits, digits, digits)
    a, b, c = triples.unbind(dim=1)
    bos = torch.full((triples.shape[0],), BOS)
    sep = torch.full((triples.shape[0],), SEP)
    eos = torch.full((triples.shape[0],), EOS)
    return torch.stack((bos, a, b, c, sep, c, b, a, eos), dim=1).long()

def split_sequences(
    sequences: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic 800/200 splits."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(sequences.shape[0], generator=rng)
    return sequences[indices[:TRAIN_SIZE]], sequences[indices[TRAIN_SIZE:]]


def make_examples(
    sequences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return inputs, targets, and boolean loss mask.

    Targets are ``[a, b, c, SEP, c, b, a, EOS]``. Loss is applied only to the
    reversed digits and EOS: ``False False False False True True True True``.
    """
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    loss_mask = torch.zeros_like(targets, dtype=torch.bool)
    loss_mask[:, 4:] = True
    return inputs, targets, loss_mask