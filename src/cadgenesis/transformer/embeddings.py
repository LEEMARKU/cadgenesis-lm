"""cadgenesis.transformer.embeddings
=================================
Reusable input embeddings for CADGenesis-LM v6.0: token, token-type and
combined input embedding building blocks used by the encoder/decoder stacks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cadgenesis.transformer.positional import SinusoidalPositionalEncoding


class TokenEmbedding(nn.Module):
    """Vocabulary token embedding with a scaling factor and optional padding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or d_model < 1:
            raise ValueError("vocab_size and d_model must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = scale if scale is not None else math.sqrt(d_model)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (..., seq_len) -> (..., seq_len, d_model)"""
        return self.embed(token_ids) * self.scale


class TypeEmbedding(nn.Module):
    """Token-family / type embedding added on top of token embeddings."""

    def __init__(self, num_types: int, d_model: int) -> None:
        super().__init__()
        if num_types < 1 or d_model < 1:
            raise ValueError("num_types and d_model must be positive")
        self.num_types = num_types
        self.embed = nn.Embedding(num_types, d_model)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        """type_ids: (..., seq_len) -> (..., seq_len, d_model)"""
        return self.embed(type_ids)


class CombinedInputEmbedding(nn.Module):
    """Token + type embeddings combined, optionally with sinusoidal positions.

    Mirrors the embedding behaviour of :class:`GeometryAwareTransformer`:
    ``pos_enc((token_embed + type_embed) * sqrt(d_model))`` when positional
    encoding is enabled.
    """

    def __init__(
        self,
        vocab_size: int,
        num_types: int,
        d_model: int,
        padding_idx: int = 0,
        max_seq_len: int = 2048,
        add_positional: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.type_embedding = TypeEmbedding(num_types, d_model)
        self.add_positional = add_positional
        self.positional = (
            SinusoidalPositionalEncoding(d_model, max_len=max_seq_len) if add_positional else None
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (..., seq_len, d_model) embeddings."""
        hidden = self.token_embedding(token_ids)
        if type_ids is not None:
            hidden = hidden + self.type_embedding(type_ids)
        if self.positional is not None:
            hidden = self.positional(hidden)
        return hidden
