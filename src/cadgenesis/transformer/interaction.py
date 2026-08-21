"""
cadgenesis.transformer.interaction
============================
Feature Interaction Layers for CADGenesis-LM v2.0.

CAD sequences are composed of *heterogeneous feature families* (geometry,
feature operations, constraints, materials, assemblies, manufacturing and
simulation tokens).  Plain self-attention lets tokens attend across the whole
sequence, but it has no explicit prior about *which* features should interact.
This module adds an optional, gated **cross-feature interaction** sub-layer to
the transformer block:

* A **type-aware interaction bias** derived from per-token family ids biases
  self-attention towards same-family / related-family tokens.
* A **feature-wise channel mixer** (MLP-Mixer style) promotes interaction
  between hidden channels within each token.
* A learned **gate** controls how strongly the interaction layer is applied per
  token, so it can be disabled by the network when unnecessary.

By default the layer is *off* (``use_feature_interaction=False``) and the block
behaves exactly like the pre-upgrade transformer — this is purely additive
capability.

Complexity
----------
    Self-interaction attention :  O(T² · C)   (only when feature ids are given)
    Channel mixer              :  O(T · C · C) (per-token MLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.positional import RotaryEmbedding

_NUM_FAMILIES = 10  # matches the 10 token family ids used by the type embedding


class FeatureInteractionLayer(nn.Module):
    """
    Gated cross-feature interaction.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    num_heads : int
        Number of heads for the type-aware self-interaction attention.
    dropout : float
        Dropout probability.
    num_families : int
        Number of token families used to index the learned interaction bias.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        num_families: int = _NUM_FAMILIES,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Type-aware interaction bias: per-family scalar offset per head.
        self.type_bias = nn.Embedding(num_families, num_heads)

        # Self-interaction attention (queries over the same sequence).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

        # Feature-wise channel mixer.
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        # Gate: how strongly to apply this layer per token.
        self.gate = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def _interaction_attention(
        self,
        x: torch.Tensor,
        feature_type_ids: torch.Tensor | None,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Type-biased self-interaction attention.
        x: (B, T, C) → (B, T, C)
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if feature_type_ids is not None:
            # bias_ij = per-query family bias (B, H, T, 1), added to all keys.
            bias = self.type_bias(feature_type_ids).transpose(1, 2).unsqueeze(-1)
            scores = scores + bias
        if causal_mask is not None:
            scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        feature_type_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        feature_type_ids: (B, T) integer family ids in [0, num_families).
        causal_mask: (1, 1, T, T) additive mask, optional.

        Returns: (B, T, C) — the input plus a gated interaction residual.
        """
        attn_out = self._interaction_attention(x, feature_type_ids, causal_mask)
        mixed = self.channel_mixer(x)

        gate = torch.sigmoid(self.gate(x))  # (B, T, 1)
        return x + gate * (attn_out + mixed)
