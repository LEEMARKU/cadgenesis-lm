"""cadgenesis.multimodal.cross_modal
===================================
Cross-modal attention (Pillar 3).

The cross-modal layer attends between *pairs* of modalities inside the
shared engineering embedding space.  It implements the eight headline
pairs from the v6 architecture:

- Text <-> CAD
- CAD <-> Images
- Sketch <-> CAD
- Drawing <-> CAD
- PointCloud <-> CAD
- Mesh <-> CAD
- Sensor <-> Simulation
- Video <-> CAD

A :class:`CrossModalLayer` is a symmetric attention block: query from one
modality, key/value from the other, residual connections on both sides, and
L2-normalised outputs so the attended tokens stay in the shared space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.embeddings import SharedEngineeringEmbeddingSpace

# The eight canonical cross-modal pairs (order-insensitive, deduplicated).
#
# ``Modality.SENSOR <-> Modality.SENSOR`` represents the Sensor <-> Simulation
# pair: the world model feeds *simulated* sensor streams back through the
# SENSOR encoder, so real-sensor embeddings attend against simulated-sensor
# embeddings (the same modality, two different sources).
HEADLINE_PAIRS: tuple[tuple[Modality, Modality], ...] = (
    (Modality.TEXT, Modality.CAD),
    (Modality.CAD, Modality.IMAGE),
    (Modality.SKETCH, Modality.CAD),
    (Modality.DRAWING, Modality.CAD),
    (Modality.POINT_CLOUD, Modality.CAD),
    (Modality.MESH, Modality.CAD),
    (Modality.SENSOR, Modality.SENSOR),
    (Modality.VIDEO, Modality.CAD),
)


def _pair_key(a: Modality, b: Modality) -> tuple[Modality, Modality]:
    return (a, b) if a <= b else (b, a)


class CrossModalLayer(nn.Module):
    """Symmetric attention between two modalities.

    Parameters
    ----------
    embed_dim : int
        Shared embedding dimension.
    num_heads : int
        Attention heads.
    dropout : float
        Dropout on attention and feed-forward.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``a``/``b``: (B, T_a, D) and (B, T_b, D).

        Returns updated (B, T_a, D) and (B, T_b, D) tensors.
        """
        attn_a, _ = self.attention(a, b, b)
        attn_b, _ = self.attention(b, a, a)
        a = self.norm_a(a + attn_a)
        b = self.norm_b(b + attn_b)
        a = self.norm_a(a + self.ffn_a(a))
        b = self.norm_b(b + self.ffn_b(b))
        return a, b


class CrossModalAttention(nn.Module):
    """Stack of :class:`CrossModalLayer` blocks over a pair of modalities.

    Applies an alternating query/key assignment across layers (odd layers
    attend A<-B, even layers B<-A) so information flows in both directions.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_seq, b_seq = a, b
        for layer in self.layers:
            a_seq, b_seq = layer(a_seq, b_seq)
        return a_seq, b_seq


class CrossModalLayerRegistry(nn.Module):
    """Registry of cross-modal stacks for many modality pairs.

    Each pair gets its own :class:`CrossModalAttention` stack, which allows
    pair-specific attention patterns (text keyed on CAD semantics, sensor
    keyed on simulation state, ...) while sharing a single embedding space.
    """

    def __init__(
        self,
        pairs: list[tuple[Modality, Modality]] | None = None,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        pairs = pairs if pairs is not None else list(HEADLINE_PAIRS)
        self._stacks: nn.ModuleDict = nn.ModuleDict()
        for pair in pairs:
            self.register_pair(pair, num_heads=num_heads, num_layers=num_layers, dropout=dropout)

    def register_pair(
        self,
        pair: tuple[Modality, Modality],
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        a, b = _pair_key(pair[0], pair[1])
        key = f"{a.value}__{b.value}"
        if key not in self._stacks:
            self._stacks[key] = CrossModalAttention(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

    def has_pair(self, a: Modality, b: Modality) -> bool:
        key = _pair_key(a, b)
        return f"{key[0].value}__{key[1].value}" in self._stacks

    def stacks(self) -> list[str]:
        return list(self._stacks)

    def forward(
        self,
        a: Modality,
        b: Modality,
        a_tokens: torch.Tensor,
        b_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend between ``a`` and ``b`` token sequences.

        ``a_tokens``/``b_tokens``: (B, T, embed_dim).  Returns the attended
        pair.  If the pair was never registered, returns the inputs unchanged.
        """
        key_a, key_b = _pair_key(a, b)
        key = f"{key_a.value}__{key_b.value}"
        if key not in self._stacks:
            return a_tokens, b_tokens
        stack = self._stacks[key]
        assert isinstance(stack, CrossModalAttention)
        if a == key_a:
            return stack(a_tokens, b_tokens)
        b_tokens_out, a_tokens_out = stack(b_tokens, a_tokens)
        return a_tokens_out, b_tokens_out


@dataclass
class CrossModalResult:
    """Output of a cross-modal attention pass."""

    a: Modality
    b: Modality
    a_attended: torch.Tensor
    b_attended: torch.Tensor
    a_pooled: torch.Tensor
    b_pooled: torch.Tensor

    def a_to_b_similarity(self) -> torch.Tensor:
        return F.cosine_similarity(self.a_pooled, self.b_pooled, dim=-1)


class CrossModalEngine(nn.Module):
    """Facade for cross-modal attention over arbitrary modality pairs.

    ``attend`` projects raw features into the shared space, runs the
    pair-specific attention stack, pools the attended token sequences and
    returns a :class:`CrossModalResult`.  Sequence-poor modalities (where
    the encoder emits a single ``(B, D)`` vector) are broadcast to a
    ``(B, 1, D)`` sequence so they can still participate.
    """

    def __init__(
        self,
        space: SharedEngineeringEmbeddingSpace,
        pairs: list[tuple[Modality, Modality]] | None = None,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.space = space
        self.embed_dim = embed_dim
        self.registry = CrossModalLayerRegistry(
            pairs=pairs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def _to_sequence(x: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, 1, D); (B, T, D) unchanged."""
        if x.dim() == 2:
            return x[:, None, :]
        return x

    def attend(
        self,
        a: Modality,
        b: Modality,
        a_features: torch.Tensor,
        b_features: torch.Tensor,
    ) -> CrossModalResult:
        """Raw features for ``a`` and ``b`` -> attended shared embeddings."""
        a_emb = self.space.embed(a, a_features)
        b_emb = self.space.embed(b, b_features)
        a_tokens = self._to_sequence(a_emb)
        b_tokens = self._to_sequence(b_emb)
        a_att, b_att = self.registry.forward(a, b, a_tokens, b_tokens)
        a_pooled = a_att.mean(dim=1)
        b_pooled = b_att.mean(dim=1)
        return CrossModalResult(
            a=a,
            b=b,
            a_attended=a_att,
            b_attended=b_att,
            a_pooled=a_pooled,
            b_pooled=b_pooled,
        )


__all__ = [
    "HEADLINE_PAIRS",
    "CrossModalAttention",
    "CrossModalEngine",
    "CrossModalLayer",
    "CrossModalLayerRegistry",
    "CrossModalResult",
]
