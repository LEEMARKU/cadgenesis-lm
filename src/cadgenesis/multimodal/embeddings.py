"""cadgenesis.multimodal.embeddings
=================================
Shared Engineering Embedding Space (Pillar 3).

Every modality encoder produces *raw features*; the shared space then:

1. runs an optional per-modality **adapter** that aligns modality-specific
   distributions,
2. projects the result through a learned **projection head** into a common
   ``embed_dim`` space,
3. applies **feature normalization** (L2 or layer norm) so that cosine
   similarity between modalities is a well-defined retrieval score.

All 11 modalities therefore live in ONE shared latent space and can be
compared, retrieved and fused directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.multimodal.common import (
    MODALITIES,
    Modality,
)

_VALID_NORMALIZATIONS = ("none", "l2", "layer_norm")


class FeatureNormalization(nn.Module):
    """Configurable feature normalization applied to projected embeddings.

    ``kind`` is one of ``"none"``, ``"l2"`` (L2 unit norm) or ``"layer_norm"``
    (learned per-channel scaling).
    """

    def __init__(self, embed_dim: int, kind: str = "l2") -> None:
        super().__init__()
        if kind not in _VALID_NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {_VALID_NORMALIZATIONS}; got {kind!r}")
        self.kind = kind
        if kind == "layer_norm":
            self.layer_norm: nn.LayerNorm | None = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return x
        if self.kind == "l2":
            return F.normalize(x, p=2, dim=-1)
        assert self.layer_norm is not None
        return self.layer_norm(x)


class ProjectionHead(nn.Module):
    """Two-layer MLP projecting raw modality features into the shared space.

    ``input_dim`` is the raw encoder feature dimension, ``output_dim`` the
    shared embedding dimension.  Outputs are normalized by the configured
    :class:`FeatureNormalization` so cross-modal cosine comparisons are
    meaningful.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        normalize: str = "l2",
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else max(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        self.normalization = FeatureNormalization(output_dim, kind=normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) -> (..., output_dim) normalized."""
        return self.normalization(self.net(x))


class ModalityAdapter(nn.Module):
    """Learned per-modality distribution adapter.

    A small MLP with a residual connection that re-scales and re-centers a
    modality's features so they land in a common distribution before the
    projection head.  When ``residual=True`` the adapter degenerates toward
    the identity for small residuals, preserving raw information.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else max(input_dim, 128)
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return x + out if self.residual else out


class ModalityAdapterRegistry(nn.Module):
    """Registry of per-modality adapters keyed by :class:`Modality`.

    Only adapters for modalities registered via :meth:`register` are instantiated;
    unknown modalities pass through unchanged (identity).
    """

    def __init__(
        self,
        adapters: dict[Modality, ModalityAdapter] | None = None,
    ) -> None:
        super().__init__()
        self._adapters: nn.ModuleDict = nn.ModuleDict()
        if adapters:
            for modality, adapter in adapters.items():
                self.register(modality, adapter)

    def register(self, modality: Modality, adapter: ModalityAdapter) -> None:
        if not isinstance(adapter, ModalityAdapter):
            raise TypeError("adapter must be a ModalityAdapter")
        self._adapters[modality.value] = adapter

    def has(self, modality: Modality) -> bool:
        return modality.value in self._adapters

    def get(self, modality: Modality) -> ModalityAdapter | None:
        if modality.value in self._adapters:
            adapter = self._adapters[modality.value]
            assert isinstance(adapter, ModalityAdapter)
            return adapter
        return None

    def adapt(self, modality: Modality, x: torch.Tensor) -> torch.Tensor:
        adapter = self.get(modality)
        return adapter(x) if adapter is not None else x

    def modalities(self) -> list[Modality]:
        return [Modality(name) for name in self._adapters]


class SharedEngineeringEmbeddingSpace(nn.Module):
    """The single shared latent space for all 11 modalities.

    ``embed`` maps raw modality features to the shared space:

    .. code-block:: python

        space = SharedEngineeringEmbeddingSpace(embed_dim=256)
        z = space.embed(Modality.CAD, cad_features)          # (B, 256)
        sim = space.similarity(z_text, z_cad)                # (B, B) cosine

    Parameters
    ----------
    embed_dim : int
        Shared embedding dimension.
    modality_input_dims : dict[Modality, int] | None
        Raw feature dim per modality.  Defaults to the canonical spec table.
    projection_hidden : int | None
        Hidden width of the projection heads.
    use_adapters : bool
        Whether to instantiate per-modality adapters.
    normalize : str
        Normalization applied to projected embeddings.
    dropout : float
        Dropout inside projection heads / adapters.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        modality_input_dims: dict[Modality, int] | None = None,
        projection_hidden: int | None = None,
        use_adapters: bool = True,
        normalize: str = "l2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if normalize not in _VALID_NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {_VALID_NORMALIZATIONS}; got {normalize!r}")
        self.embed_dim = embed_dim
        self.normalize = normalize

        input_dims = _default_input_dims()
        if modality_input_dims:
            input_dims.update({m: int(d) for m, d in modality_input_dims.items()})
        self.input_dims: dict[Modality, int] = input_dims

        # One projection head per modality into the shared space.
        heads: dict[str, nn.Module] = {}
        for modality in MODALITIES:
            heads[modality.value] = ProjectionHead(
                input_dim=input_dims[modality],
                output_dim=embed_dim,
                hidden_dim=projection_hidden,
                dropout=dropout,
                normalize="none",  # normalization applied once, after adapters
            )
        self.projection_heads = nn.ModuleDict(heads)

        # Optional per-modality adapters.
        self.use_adapters = use_adapters
        if use_adapters:
            adapters = {
                modality: ModalityAdapter(
                    input_dim=embed_dim,
                    hidden_dim=max(embed_dim // 2, 32),
                    dropout=dropout,
                )
                for modality in MODALITIES
            }
        else:
            adapters = {}
        self.adapter_registry = ModalityAdapterRegistry(adapters)

        # Shared final normalization applied to every modality's embedding.
        self.normalization = FeatureNormalization(embed_dim, kind=normalize)

    # ------------------------------------------------------------------ embed

    def _project(self, modality: Modality, x: torch.Tensor) -> torch.Tensor:
        if modality.value not in self.projection_heads:
            raise KeyError(f"no projection head for modality {modality.value!r}")
        head = self.projection_heads[modality.value]
        z = head(x)
        return self.adapter_registry.adapt(modality, z)

    def embed(self, modality: Modality, features: torch.Tensor) -> torch.Tensor:
        """Project raw ``features`` (..., input_dim[modality]) -> (..., embed_dim)."""
        if features.shape[-1] != self.input_dims[modality]:
            raise ValueError(
                f"features for {modality.value!r} have last dim "
                f"{features.shape[-1]} but expected {self.input_dims[modality]}"
            )
        return self.normalization(self._project(modality, features))

    def embed_many(
        self,
        features: dict[Modality, torch.Tensor],
    ) -> dict[Modality, torch.Tensor]:
        """Project a dict of per-modality raw features into the shared space."""
        return {modality: self.embed(modality, feats) for modality, feats in features.items()}

    # ---------------------------------------------------------------- metrics

    @torch.no_grad()
    def similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cosine similarity between embedding matrices.

        ``a``/``b`` are (B, D) or (N, D); when ``b`` is None, ``a`` is compared
        against itself.  Embeddings are L2-normalized on the fly so the cosine
        is exact even if ``normalize`` is off.
        """
        a_n = F.normalize(a, p=2, dim=-1)
        b_n = F.normalize(b, p=2, dim=-1) if b is not None else a_n
        return a_n @ b_n.transpose(-1, -2)

    @torch.no_grad()
    def cross_modal_similarity(
        self,
        features: dict[Modality, torch.Tensor],
    ) -> dict[tuple[Modality, Modality], torch.Tensor]:
        """Pairwise similarity between every pair of modality embeddings."""
        embeds = self.embed_many(features)
        result: dict[tuple[Modality, Modality], torch.Tensor] = {}
        names = list(embeds)
        for i, ma in enumerate(names):
            for mb in names[i:]:
                result[(ma, mb)] = self.similarity(embeds[ma], embeds[mb])
        return result


def _default_input_dims() -> dict[Modality, int]:
    from cadgenesis.multimodal.common import DEFAULT_FEATURE_DIMS

    return dict(DEFAULT_FEATURE_DIMS)


__all__ = [
    "FeatureNormalization",
    "ModalityAdapter",
    "ModalityAdapterRegistry",
    "ProjectionHead",
    "SharedEngineeringEmbeddingSpace",
]
