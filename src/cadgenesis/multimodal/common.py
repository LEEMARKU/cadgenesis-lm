"""cadgenesis.multimodal.common
=============================
Shared vocabulary of the Multimodal Understanding subsystem (Pillar 3).

Defines the :class:`Modality` enumeration covering every modality the unified
multimodal foundation model understands, the modality capability registry
(which sub-modalities / file families each modality supports), and a small
:class:`ModalitySpec` record used by the encoder registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class Modality(str, Enum):
    """Every modality projected into the shared engineering embedding space.

    Members are ordered by the number of token/feature families each carries;
    the ordering is stable and used for deterministic weight-matrix naming.
    """

    TEXT = "text"
    CAD = "cad"
    DRAWING = "drawing"
    SKETCH = "sketch"
    IMAGE = "image"
    PDF = "pdf"
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"

    def __str__(self) -> str:
        return self.value


# All supported modalities in canonical order.
MODALITIES: tuple[Modality, ...] = tuple(Modality)

# Public alias so callers can iterate over the full modality set.
ALL_MODALITIES: tuple[Modality, ...] = MODALITIES


@dataclass
class ModalitySpec:
    """Static description of one modality.

    Attributes
    ----------
    modality : Modality
        The canonical modality.
    supports : tuple[str, ...]
        The concrete input families / file formats the modality understands
        (engineering prompts, STEP/IGES/... CAD files, LiDAR point clouds, ...).
    feature_dim : int
        Dimension of the encoder output fed to the shared projection head.
    sequence_aware : bool
        True when the encoder may emit more than one latent token (e.g. a
        per-frame video feature, per-page PDF feature) instead of a single
        pooled vector.
    """

    modality: Modality
    supports: tuple[str, ...]
    feature_dim: int
    sequence_aware: bool = False

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "supports": list(self.supports),
            "feature_dim": self.feature_dim,
            "sequence_aware": self.sequence_aware,
        }


# Canonical capability registry: modality -> (supported families, feature dim,
# sequence-aware flag).  Feature dims are configurable at construction time via
# ``MultimodalConfig``; these defaults are used when no config is supplied.
_MODALITY_SUPPORTS: dict[Modality, tuple[tuple[str, ...], int, bool]] = {
    Modality.TEXT: (
        (
            "engineering_prompts",
            "conversational_reasoning",
            "technical_terminology",
            "engineering_specifications",
        ),
        512,
        False,
    ),
    Modality.CAD: (
        (
            "STEP",
            "IGES",
            "PARASOLID",
            "FUSION360",
            "SOLIDWORKS",
            "FREECAD",
            "OPENSCAD",
        ),
        384,
        False,
    ),
    Modality.DRAWING: (
        (
            "dimensions",
            "annotations",
            "title_blocks",
            "symbols",
            "section_views",
            "exploded_views",
        ),
        256,
        False,
    ),
    Modality.SKETCH: (
        (
            "hand_sketches",
            "digital_sketches",
            "construction_lines",
            "dimensions",
            "constraints",
        ),
        256,
        False,
    ),
    Modality.IMAGE: (
        (
            "product_photos",
            "cad_screenshots",
            "rendered_models",
            "manufacturing_images",
        ),
        256,
        False,
    ),
    Modality.PDF: (
        (
            "engineering_manuals",
            "specifications",
            "standards",
            "technical_reports",
        ),
        384,
        True,
    ),
    Modality.POINT_CLOUD: (("LiDAR", "structured_scans", "RGB-D"), 256, False),
    Modality.MESH: (("STL", "OBJ", "GLTF", "PLY"), 256, False),
    Modality.AUDIO: (
        (
            "speech_recognition",
            "engineering_commands",
            "design_discussions",
        ),
        256,
        True,
    ),
    Modality.VIDEO: (
        (
            "assembly_videos",
            "manufacturing_videos",
            "instructional_videos",
        ),
        256,
        True,
    ),
    Modality.SENSOR: (
        (
            "vibration",
            "force",
            "temperature",
            "pressure",
            "telemetry",
        ),
        256,
        True,
    ),
}

# Default per-modality feature dims (used when no ``MultimodalConfig`` present).
DEFAULT_FEATURE_DIMS: dict[Modality, int] = {
    modality: dim for modality, (_, dim, _) in _MODALITY_SUPPORTS.items()
}


def modality_specs(
    feature_dims: dict[Modality, int] | None = None,
) -> list[ModalitySpec]:
    """Build the canonical ``ModalitySpec`` list for every modality.

    ``feature_dims`` optionally overrides the default per-modality raw feature
    dimensions (e.g. from ``MultimodalConfig``).
    """
    dims = dict(DEFAULT_FEATURE_DIMS)
    if feature_dims:
        for modality, dim in feature_dims.items():
            if modality in dims:
                dims[modality] = int(dim)
    specs = []
    for modality, (supports, _, sequence_aware) in _MODALITY_SUPPORTS.items():
        specs.append(
            ModalitySpec(
                modality=modality,
                supports=supports,
                feature_dim=dims[modality],
                sequence_aware=sequence_aware,
            )
        )
    return specs


def modality_from_name(name: str) -> Modality:
    """Resolve a ``Modality`` from its string value (case-insensitive)."""
    try:
        return Modality(name.lower())
    except ValueError:
        raise KeyError(
            f"unknown modality {name!r}; expected one of {[m.value for m in MODALITIES]}"
        ) from None


@dataclass
class _CapabilityRegistry:
    """Internal registry keeping every capability family known per modality."""

    specs: ClassVar[list[ModalitySpec]] = None  # type: ignore[assignment]

    @classmethod
    def families(cls, modality: Modality) -> tuple[str, ...]:
        return next(
            (spec.supports for spec in cls.specs if spec.modality is modality),
            (),
        )

    @classmethod
    def feature_dim(cls, modality: Modality) -> int:
        return next(
            (spec.feature_dim for spec in cls.specs if spec.modality is modality),
            0,
        )


_CapabilityRegistry.specs = list(modality_specs())


def supported_families(modality: Modality) -> tuple[str, ...]:
    """Return the supported input families for ``modality``."""
    return _CapabilityRegistry.families(modality)


__all__ = [
    "ALL_MODALITIES",
    "DEFAULT_FEATURE_DIMS",
    "MODALITIES",
    "Modality",
    "ModalitySpec",
    "modality_from_name",
    "modality_specs",
    "supported_families",
]
