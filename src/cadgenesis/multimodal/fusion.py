"""cadgenesis.multimodal.fusion
==============================
Multi-modal fusion (Pillar 3).

Five fusion strategies are provided, each consuming a ``dict[Modality,
torch.Tensor]`` of *shared-space* embeddings and returning a single fused
representation:

- ``early``       — concatenation then one linear projection.
- ``late``        — weighted mean of per-modality outputs.
- ``hierarchical``— modality *families* (geometry / document / perception /
  sensory / sequence) are fused pairwise, then combined across families.
- ``adaptive``    — gated combination; per-modality gates learn which
  modalities are informative for the current input.
- ``attention``   — a multi-head attention layer over the modality tokens
  (learned modality-order-invariant pooling).

All strategies are wrapped by the :class:`FusionEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import MODALITIES, Modality

_FAMILY_GROUPS: dict[str, list[Modality]] = {
    "geometry": [Modality.CAD, Modality.MESH, Modality.POINT_CLOUD],
    "document": [Modality.TEXT, Modality.DRAWING, Modality.PDF],
    "perception": [Modality.IMAGE, Modality.SKETCH, Modality.VIDEO],
    "sensory": [Modality.AUDIO, Modality.SENSOR],
    "sequence": [],
}


class FusionStrategy(str, Enum):
    """Supported fusion strategy names."""

    EARLY = "early"
    LATE = "late"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    ATTENTION = "attention"


@dataclass
class FusionResult:
    """Fused embedding plus diagnostics."""

    fused: torch.Tensor
    strategy: FusionStrategy
    modality_weights: dict[Modality, torch.Tensor] = field(default_factory=dict)
    family_representations: dict[str, torch.Tensor] = field(default_factory=dict)


class _EarlyFusion(nn.Module):
    """Concatenate all modality embeddings, then project to ``out_dim``."""

    def __init__(self, embed_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(embed_dim * len(MODALITIES), out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, embeddings: dict[Modality, torch.Tensor]) -> torch.Tensor:
        present = [m for m in MODALITIES if m in embeddings]
        if not present:
            raise ValueError("early fusion requires at least one modality")
        batch = embeddings[present[0]].shape[0]
        device = embeddings[present[0]].device
        pieces: list[torch.Tensor] = []
        for modality in MODALITIES:
            if modality in embeddings:
                tensor = embeddings[modality]
                pieces.append(tensor if tensor.dim() == 2 else tensor.mean(dim=1))
            else:
                pieces.append(torch.zeros(batch, self.embed_dim, device=device))
        features = torch.cat(pieces, dim=-1)
        return self.projector(features)


class _LateFusion(nn.Module):
    """Learned-weight mean of per-modality embeddings."""

    def __init__(self, embed_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.logits = nn.Parameter(torch.zeros(len(MODALITIES)))
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weights: dict[Modality, torch.Tensor] = {}

    def forward(self, embeddings: dict[Modality, torch.Tensor]) -> torch.Tensor:
        present = [m for m in MODALITIES if m in embeddings]
        if not present:
            raise ValueError("late fusion requires at least one modality")
        indices = torch.tensor([MODALITIES.index(m) for m in present], dtype=torch.long)
        weights = torch.softmax(self.logits[indices], dim=0)
        batch = embeddings[present[0]].shape[0]
        summed = torch.zeros(batch, self.embed_dim, device=weights.device)
        for weight, modality in zip(weights, present, strict=True):
            tensor = embeddings[modality]
            if tensor.dim() != 2:
                tensor = tensor.mean(dim=1)
            summed = summed + weight * tensor
        self.weights = {m: weights[i].detach().clone().squeeze() for i, m in enumerate(present)}
        return self.projector(summed)


class _HierarchicalFusion(nn.Module):
    """Fuse within modality families, then across families."""

    def __init__(self, embed_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.family_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in _FAMILY_GROUPS
            }
        )
        self.final = nn.Sequential(
            nn.Linear(embed_dim * len(_FAMILY_GROUPS), out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.family_reps: dict[str, torch.Tensor] = {}

    def forward(self, embeddings: dict[Modality, torch.Tensor]) -> torch.Tensor:
        family_reps: dict[str, torch.Tensor] = {}
        present = [m for m in MODALITIES if m in embeddings]
        if not present:
            raise ValueError("hierarchical fusion requires at least one modality")
        device = embeddings[present[0]].device
        batch = embeddings[present[0]].shape[0]
        for family, members in _FAMILY_GROUPS.items():
            tensors = [
                (t if t.dim() == 2 else t.mean(dim=1))
                for m in members
                if m in embeddings
                for t in [embeddings[m]]
            ]
            if tensors:
                stacked = torch.stack(tensors, dim=0).mean(dim=0)
                family_reps[family] = self.family_projectors[family](stacked)
            else:
                family_reps[family] = torch.zeros(batch, self.embed_dim, device=device)
        self.family_reps = family_reps
        rep_list = [family_reps[family] for family in _FAMILY_GROUPS]
        return self.final(torch.cat(rep_list, dim=-1))


class _AdaptiveFusion(nn.Module):
    """Gated combination: sigmoid gates select informative modalities."""

    def __init__(self, embed_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.gates = nn.ModuleDict({m.value: nn.Linear(embed_dim, 1) for m in MODALITIES})
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weights: dict[Modality, torch.Tensor] = {}

    def forward(self, embeddings: dict[Modality, torch.Tensor]) -> torch.Tensor:
        present = [m for m in MODALITIES if m in embeddings]
        if not present:
            raise ValueError("adaptive fusion requires at least one modality")
        gate_values: list[torch.Tensor] = []
        for modality in present:
            tensor = embeddings[modality]
            if tensor.dim() != 2:
                tensor = tensor.mean(dim=1)
            gate_values.append(torch.sigmoid(self.gates[modality.value](tensor)))
        gates = torch.stack(gate_values, dim=0)
        gates = gates / gates.sum(dim=0, keepdim=True).clamp_min(1e-8)
        self.weights = {m: gates[i].detach().clone().squeeze() for i, m in enumerate(present)}
        acc = None
        for gate, modality in zip(gates, present, strict=True):
            tensor = embeddings[modality]
            if tensor.dim() != 2:
                tensor = tensor.mean(dim=1)
            weighted = gate.squeeze(-1)[:, None] * tensor
            acc = weighted if acc is None else acc + weighted
        return self.projector(acc)


class _AttentionFusion(nn.Module):
    """Multi-head attention over modality tokens (order-invariant pooling)."""

    def __init__(self, embed_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weights: dict[Modality, torch.Tensor] = {}

    def forward(self, embeddings: dict[Modality, torch.Tensor]) -> torch.Tensor:
        present = [m for m in MODALITIES if m in embeddings]
        if not present:
            raise ValueError("attention fusion requires at least one modality")
        tokens = torch.stack(
            [
                (embeddings[m] if embeddings[m].dim() == 2 else embeddings[m].mean(dim=1))
                for m in present
            ],
            dim=1,
        )  # (B, n_modalities, D)
        query = tokens.mean(dim=1, keepdim=True)
        attended, attn_weights = self.attention(query, tokens, tokens)
        attended = self.norm(query + attended)
        self.weights = {m: attn_weights[0, 0, i].detach().clone() for i, m in enumerate(present)}
        return self.projector(attended.squeeze(1))


class FusionEngine(nn.Module):
    """Dispatchable multi-modal fusion.

    Parameters
    ----------
    strategy : FusionStrategy | str
        One of the five strategies.
    embed_dim : int
        Shared embedding dimension (per-modality vectors before fusion).
    out_dim : int
        Dimension of the fused representation (defaults to ``embed_dim``).
    """

    def __init__(
        self,
        strategy: FusionStrategy | str = FusionStrategy.ATTENTION,
        embed_dim: int = 256,
        out_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(strategy, FusionStrategy):
            self.strategy = strategy
        else:
            self.strategy = FusionStrategy(strategy)
        self.embed_dim = embed_dim
        self.out_dim = out_dim if out_dim is not None else embed_dim
        if self.strategy is FusionStrategy.EARLY:
            self._impl: nn.Module = _EarlyFusion(self.embed_dim, self.out_dim, dropout)
        elif self.strategy is FusionStrategy.LATE:
            self._impl = _LateFusion(self.embed_dim, self.out_dim, dropout)
        elif self.strategy is FusionStrategy.HIERARCHICAL:
            self._impl = _HierarchicalFusion(self.embed_dim, self.out_dim, dropout)
        elif self.strategy is FusionStrategy.ADAPTIVE:
            self._impl = _AdaptiveFusion(self.embed_dim, self.out_dim, dropout)
        elif self.strategy is FusionStrategy.ATTENTION:
            self._impl = _AttentionFusion(self.embed_dim, self.out_dim, dropout)
        else:
            raise ValueError(f"unknown fusion strategy {self.strategy!r}")

    def forward(
        self,
        embeddings: dict[Modality, torch.Tensor],
    ) -> FusionResult:
        """Fuse a dict of shared-space embeddings into one vector per sample."""
        fused = self._impl(embeddings)
        weights: dict[Modality, torch.Tensor] = {}
        families: dict[str, torch.Tensor] = {}
        impl = self._impl
        if isinstance(impl, (_LateFusion, _AdaptiveFusion, _AttentionFusion)):
            weights = impl.weights
        elif isinstance(impl, _HierarchicalFusion):
            families = impl.family_reps
        return FusionResult(
            fused=fused,
            strategy=self.strategy,
            modality_weights=weights,
            family_representations=families,
        )


__all__ = [
    "FusionEngine",
    "FusionResult",
    "FusionStrategy",
]
