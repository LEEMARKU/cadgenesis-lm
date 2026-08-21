"""cadgenesis.multimodal.encoders.drawing
========================================
Engineering-drawing encoder (dimensions, annotations, title blocks, symbols,
section views, exploded views).

A drawing is normalised into a :class:`DrawingDocument`: the title-block
fields, a list of :class:`DrawingDimension` records, annotations, symbols,
and the drawing views (each tagged with a view family — section / exploded /
orthographic / isometric / detail).  A deterministic structural descriptor
captures the drawing's content, and an MLP maps it into the shared raw
feature space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder

_VIEW_FAMILIES: tuple[str, ...] = (
    "section",
    "exploded",
    "orthographic",
    "isometric",
    "detail",
)

_DIMENSION_KINDS: tuple[str, ...] = (
    "linear",
    "angular",
    "radius",
    "diameter",
    "ordinate",
    "reference",
)

_SYMBOL_KINDS: tuple[str, ...] = (
    "surface_finish",
    "weld",
    "datum",
    "feature_control_frame",
    "centerline",
    "cutting_plane",
    "section_hatch",
    "balloon",
)


@dataclass
class DrawingDimension:
    """A dimension annotation on a drawing."""

    kind: str
    value: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _DIMENSION_KINDS:
            raise ValueError(
                f"unknown dimension kind {self.kind!r}; expected one of {_DIMENSION_KINDS}"
            )


@dataclass
class DrawingView:
    """A single view on an engineering drawing."""

    name: str
    family: str
    entities: int = 0

    def __post_init__(self) -> None:
        if self.family not in _VIEW_FAMILIES:
            raise ValueError(
                f"unknown view family {self.family!r}; expected one of {_VIEW_FAMILIES}"
            )


@dataclass
class DrawingDocument:
    """Normalised representation of an engineering drawing."""

    name: str = ""
    title_block: dict[str, str] = field(default_factory=dict)
    dimensions: list[DrawingDimension] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    views: list[DrawingView] = field(default_factory=list)

    def add_dimension(self, kind: str, value: float, name: str = "") -> DrawingDimension:
        dimension = DrawingDimension(kind=kind, value=value, name=name)
        self.dimensions.append(dimension)
        return dimension

    def add_view(self, name: str, family: str, entities: int = 0) -> DrawingView:
        view = DrawingView(name=name, family=family, entities=entities)
        self.views.append(view)
        return view


_DESCRIPTOR_SIZE = (
    len(_VIEW_FAMILIES)
    + len(_DIMENSION_KINDS)
    + len(_SYMBOL_KINDS)
    + 10  # annotations, title-block fields, symbol count, stats, ...
)


def drawing_document_descriptor(document: DrawingDocument) -> torch.Tensor:
    """Deterministic structural descriptor of an engineering drawing."""
    vec = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    offset = 0

    for view in document.views:
        vec[offset + _VIEW_FAMILIES.index(view.family)] += 1.0
    vec[offset : offset + len(_VIEW_FAMILIES)] /= max(len(document.views), 1)
    offset += len(_VIEW_FAMILIES)

    values: list[float] = []
    for dimension in document.dimensions:
        vec[offset + _DIMENSION_KINDS.index(dimension.kind)] += 1.0
        values.append(abs(dimension.value))
    vec[offset : offset + len(_DIMENSION_KINDS)] /= max(len(document.dimensions), 1)
    offset += len(_DIMENSION_KINDS)

    for symbol in document.symbols:
        lowered = symbol.lower()
        for i, kind in enumerate(_SYMBOL_KINDS):
            if kind.replace("_", " ") in lowered or kind in lowered:
                vec[offset + i] += 1.0
    vec[offset : offset + len(_SYMBOL_KINDS)] /= max(len(document.symbols), 1)
    offset += len(_SYMBOL_KINDS)

    vec[offset + 0] = math.log1p(len(document.annotations))
    vec[offset + 1] = math.log1p(len(document.title_block))
    vec[offset + 2] = math.log1p(len(document.symbols))
    vec[offset + 3] = math.log1p(sum(v.entities for v in document.views))
    vec[offset + 4] = 1.0 if document.title_block.get("title") else 0.0
    vec[offset + 5] = 1.0 if document.title_block.get("sheet") else 0.0
    vec[offset + 6] = sum(values) if values else 0.0
    vec[offset + 7] = (sum(values) / len(values)) if values else 0.0
    vec[offset + 8] = max(values) if values else 0.0
    vec[offset + 9] = len({a.lower() for a in document.annotations}) / max(
        len(document.annotations), 1
    )
    return vec


class DrawingEncoder(MultimodalEncoder):
    """Encoder for the ``drawing`` modality."""

    modality: ClassVar[Modality] = Modality.DRAWING

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.net = nn.Sequential(
            nn.Linear(_DESCRIPTOR_SIZE, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, descriptor_size) -> (B, feature_dim)."""
        if x.dim() != 2 or x.shape[-1] != _DESCRIPTOR_SIZE:
            raise ValueError(
                f"drawing encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, DrawingDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty drawing batch")
        descriptors = torch.stack([drawing_document_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = [
    "DrawingDimension",
    "DrawingDocument",
    "DrawingEncoder",
    "DrawingView",
    "drawing_document_descriptor",
]
