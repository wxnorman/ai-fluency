import math

import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        context_length: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_length = context_length
        
        # One projection calculates Q, K, and V together.
        self.qkv_projection = nn.Linear(
            d_model,
            3 * d_model,
        )

        # Used after all attention heads are merged.
        self.output_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        mask = mask.view(
            1,
            1,
            context_length,
            context_length,
        )

        self.register_buffer("causal_mask", mask)

    def project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [B, T, C], got {x.shape}"
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected embedding dimension {self.d_model}, "
                f"got {x.shape[-1]}"
            )

        combined_qkv = self.qkv_projection(x)

        q, k, v = combined_qkv.chunk(3, dim=-1)

        return self.split_heads(q), self.split_heads(k), self.split_heads(v)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, time, channels = tensor.shape

        if channels != self.d_model:
            raise ValueError(
                f"Expected {self.d_model} channels, got {channels}"
            )

        tensor = tensor.reshape(
            batch_size,
            time,
            self.n_heads,
            self.head_dim,
        )

        return tensor.transpose(1, 2)

    def compute_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        if q.shape != k.shape:
            raise ValueError(
                f"Q and K must have matching shapes, got {q.shape} and {k.shape}"
            )

        if q.ndim != 4:
            raise ValueError(
                f"Expected Q and K with shape [B, H, T, D], got {q.shape}"
            )

        if q.shape[1] != self.n_heads:
            raise ValueError(
                f"Expected {self.n_heads} heads, got {q.shape[1]}"
            )

        if q.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head dimension {self.head_dim}, got {q.shape[-1]}"
            )

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        return scores

    def compute_attention_probabilities(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if scores.ndim != 4:
            raise ValueError(
                f"Expected scores with shape [B, H, T, T], got {scores.shape}"
            )

        _, n_heads, query_length, key_length = scores.shape

        if n_heads != self.n_heads:
            raise ValueError(
                f"Expected {self.n_heads} heads, got {n_heads}"
            )

        if query_length != key_length:
            raise ValueError(
                "Self-attention scores must have equal query and key lengths"
            )

        if query_length > self.context_length:
            raise ValueError(
                f"Sequence length {query_length} exceeds "
                f"context length {self.context_length}"
            )

        mask = self.causal_mask[
            :,
            :,
            :query_length,
            :key_length,
        ]

        masked_scores = scores.masked_fill(
            ~mask,
            -torch.inf,
        )

        probabilities = torch.softmax(
            masked_scores,
            dim=-1,
        )

        return probabilities

    def merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected shape [B, H, T, D], got {tensor.shape}"
            )

        batch_size, n_heads, time, head_dim = tensor.shape

        if n_heads != self.n_heads:
            raise ValueError(
                f"Expected {self.n_heads} heads, got {n_heads}"
            )

        if head_dim != self.head_dim:
            raise ValueError(
                f"Expected head dimension {self.head_dim}, got {head_dim}"
            )

        tensor = tensor.transpose(1, 2).contiguous()

        return tensor.view(
            batch_size,
            time,
            self.d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        if x.shape[1] > self.context_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds "
                f"context length {self.context_length}"
            )

        q, k, v = self.project_qkv(x)

        scores = self.compute_attention_scores(q, k)

        probabilities = self.compute_attention_probabilities(scores)

        dropped_probabilities = self.attention_dropout(
            probabilities
        )

        context = dropped_probabilities @ v

        merged = self.merge_heads(context)

        output = self.output_projection(merged)
        output = self.output_dropout(output)

        if return_attention:
            return output, probabilities

        return output


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        context_length: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=dropout,
        )

        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        normalized = self.attention_norm(x)

        if return_attention:
            attention_output, probabilities = self.attention(
                normalized,
                return_attention=True,
            )
        else:
            attention_output = self.attention(normalized)

        x = x + attention_output

        normalized = self.feed_forward_norm(x)
        feed_forward_output = self.feed_forward(normalized)

        x = x + feed_forward_output

        if return_attention:
            return x, probabilities

        return x