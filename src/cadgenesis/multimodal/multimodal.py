"""cadgenesis.multimodal.multimodal
==================================
MultimodalSystem — the Pillar 3 facade.

The facade wires together the four Pillar 3 sub-systems:

* :class:`~cadgenesis.multimodal.embeddings.SharedEngineeringEmbeddingSpace`
* the eleven modality encoders
* the cross-modal attention engine
* the fusion engine

``MultimodalSystem.encode`` accepts a dict of ``{Modality: raw input}`` and
returns a :class:`MultimodalEncoding` holding the raw per-modality features,
the shared-space embeddings, the fused representation and a cross-modal
similarity matrix.  It is trainable end-to-end (the projection heads and
fusion weights update when the system is used inside a loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import ALL_MODALITIES, Modality
from cadgenesis.multimodal.cross_modal import CrossModalEngine
from cadgenesis.multimodal.embeddings import SharedEngineeringEmbeddingSpace
from cadgenesis.multimodal.encoders import build_encoders
from cadgenesis.multimodal.encoders.base import MultimodalEncoder
from cadgenesis.multimodal.fusion import FusionEngine, FusionStrategy


@dataclass
class MultimodalEncoding:
    """Output of a :class:`MultimodalSystem` encode pass."""

    raw_features: dict[Modality, torch.Tensor] = field(default_factory=dict)
    embeddings: dict[Modality, torch.Tensor] = field(default_factory=dict)
    fused: torch.Tensor | None = None
    cross_modal_similarity: dict[tuple[Modality, Modality], torch.Tensor] = field(
        default_factory=dict
    )


class MultimodalSystem(nn.Module):
    """End-to-end multimodal understanding for all 11 modalities.

    Parameters
    ----------
    embed_dim : int
        Shared engineering embedding dimension.
    feature_dims : dict[str, int] | None
        Per-modality raw feature dims keyed by modality name (defaults to the
        canonical spec table).  Values override the defaults.
    fusion_strategy : FusionStrategy | str
        One of ``early``/``late``/``hierarchical``/``adaptive``/``attention``.
    cross_modal_heads / cross_modal_layers : int
        Attention geometry of the cross-modal stacks.
    projection_hidden : int | None
        Hidden width of the shared-space projection heads.
    use_modality_adapters : bool
        Enable per-modality distribution adapters.
    normalize : str
        Embedding normalization ("none" | "l2" | "layer_norm").
    dropout : float
        Dropout throughout.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        feature_dims: dict[str, int] | None = None,
        fusion_strategy: FusionStrategy | str = FusionStrategy.ATTENTION,
        cross_modal_heads: int = 4,
        cross_modal_layers: int = 2,
        projection_hidden: int | None = None,
        use_modality_adapters: bool = True,
        normalize: str = "l2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Normalize feature-dims keyed by name into the {Modality: int} form.
        name_dims: dict[Modality, int] = {}
        if feature_dims:
            for name, dim in feature_dims.items():
                modality = Modality(name)
                name_dims[modality] = dim
        self.feature_dims = name_dims

        self.encoders: nn.ModuleDict = nn.ModuleDict()
        built = build_encoders(feature_dims=name_dims, dropout=dropout)
        for modality, encoder in built.items():
            self.encoders[modality.value] = encoder

        self.space = SharedEngineeringEmbeddingSpace(
            embed_dim=embed_dim,
            modality_input_dims=self.raw_feature_dims(),
            projection_hidden=projection_hidden,
            use_adapters=use_modality_adapters,
            normalize=normalize,
            dropout=dropout,
        )
        self.cross_modal = CrossModalEngine(
            space=self.space,
            embed_dim=embed_dim,
            num_heads=cross_modal_heads,
            num_layers=cross_modal_layers,
            dropout=dropout,
        )
        self.fusion = FusionEngine(
            strategy=fusion_strategy,
            embed_dim=embed_dim,
            dropout=dropout,
        )

    # ---------------------------------------------------------------- access

    def raw_feature_dims(self) -> dict[Modality, int]:
        return {
            Modality(name): encoder.feature_dim
            for name, encoder in self.encoders.items()
            if isinstance(encoder, MultimodalEncoder)
        }

    def get_encoder(self, modality: Modality) -> MultimodalEncoder:
        encoder = self.encoders[modality.value]
        assert isinstance(encoder, MultimodalEncoder)
        return encoder

    def encode_modality(self, modality: Modality, inputs: Any) -> torch.Tensor:
        """Encode ``inputs`` for one modality into raw features."""
        encoder = self.get_encoder(modality)
        return encoder.encode(inputs)

    def embed_modality(self, modality: Modality, inputs: Any) -> torch.Tensor:
        """Encode + project one modality into the shared embedding space."""
        return self.space.embed(modality, self.encode_modality(modality, inputs))

    # ------------------------------------------------------------------ encode

    def encode(self, inputs: dict[Modality, Any]) -> MultimodalEncoding:
        """Encode a ``{modality: input}`` dict.

        Returns raw features, shared-space embeddings, a fused representation
        (when at least one modality is present) and the pairwise cross-modal
        cosine-similarity matrix.
        """
        if not inputs:
            raise ValueError("multimodal encode requires at least one modality")

        raw: dict[Modality, torch.Tensor] = {}
        embeddings: dict[Modality, torch.Tensor] = {}
        for modality, data in inputs.items():
            features = self.encode_modality(modality, data)
            raw[modality] = features
            embeddings[modality] = self.space.embed(modality, features)

        fused = self.fusion(embeddings).fused

        names = list(embeddings)
        similarity: dict[tuple[Modality, Modality], torch.Tensor] = {}
        for i, ma in enumerate(names):
            for mb in names[i:]:
                similarity[(ma, mb)] = self.space.similarity(embeddings[ma], embeddings[mb])
        return MultimodalEncoding(
            raw_features=raw,
            embeddings=embeddings,
            fused=fused,
            cross_modal_similarity=similarity,
        )

    def encode_cross_modal(
        self,
        a: Modality,
        b: Modality,
        a_inputs: Any,
        b_inputs: Any,
    ) -> Any:
        """Run pairwise cross-modal attention between two modalities."""
        return self.cross_modal.attend(
            a,
            b,
            self.encode_modality(a, a_inputs),
            self.encode_modality(b, b_inputs),
        )

    # ------------------------------------------------------------ convenience

    @classmethod
    def from_config(cls, config: Any) -> MultimodalSystem:
        """Build a system from a ``MultimodalConfig``-shaped object."""
        return cls(
            embed_dim=config.embed_dim,
            feature_dims=config.feature_dims(),
            fusion_strategy=config.fusion_strategy,
            cross_modal_heads=config.cross_modal_heads,
            cross_modal_layers=config.cross_modal_layers,
            projection_hidden=config.projection_hidden,
            use_modality_adapters=config.use_modality_adapters,
            normalize=config.normalize,
            dropout=config.dropout,
        )

    def modality_names(self) -> list[str]:
        return [m.value for m in ALL_MODALITIES]


__all__ = [
    "MultimodalEncoding",
    "MultimodalSystem",
]
