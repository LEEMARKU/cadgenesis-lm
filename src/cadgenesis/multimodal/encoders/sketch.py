"""cadgenesis.multimodal.encoders.sketch
======================================
Sketch encoder (hand sketches, digital sketches, construction lines,
dimensions, constraints).

A sketch is normalised into a :class:`SketchDocument` — the flat list of
sketch entities (points, lines, arcs, circles, splines), the construction
entities, dimensional constraints and geometric constraints, plus the 2D
bounds of the profile.  The encoder builds a structural descriptor from
entity-family histograms, constraint counts and closed-profile statistics.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder

_ENTITY_KINDS: tuple[str, ...] = ("point", "line", "arc", "circle", "spline")
_CONSTRAINT_KINDS: tuple[str, ...] = (
    "coincident",
    "collinear",
    "parallel",
    "perpendicular",
    "tangent",
    "horizontal",
    "vertical",
    "equal",
    "concentric",
    "fixed",
    "midpoint",
    "symmetric",
)
_DIMENSION_KINDS: tuple[str, ...] = ("linear", "angular", "radius", "diameter")


@dataclass
class SketchEntity:
    """A sketch entity (kind + construction flag + defining points)."""

    kind: str
    points: list[tuple[float, float]] = field(default_factory=list)
    is_construction: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _ENTITY_KINDS:
            raise ValueError(
                f"unknown sketch entity kind {self.kind!r}; expected one of {_ENTITY_KINDS}"
            )


@dataclass
class SketchDocument:
    """Normalised representation of a 2D parametric sketch."""

    name: str = ""
    entities: list[SketchEntity] = field(default_factory=list)
    dimensions: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    source: str = "digital"  # "hand" | "digital"

    def add_entity(
        self,
        kind: str,
        points: list[tuple[float, float]] | None = None,
        is_construction: bool = False,
        name: str = "",
    ) -> SketchEntity:
        entity = SketchEntity(
            kind=kind,
            points=list(points or []),
            is_construction=is_construction,
            name=name,
        )
        self.entities.append(entity)
        return entity

    @property
    def is_closed(self) -> bool:
        """Best-effort closed-profile heuristic: at least 3 entities forming a
        closed chain when each non-construction entity is a single segment."""
        segments = [e for e in self.entities if not e.is_construction]
        if len(segments) < 3:
            return False
        endpoints: dict[tuple[float, float], int] = {}
        for entity in segments:
            pts = entity.points
            if len(pts) < 2:
                return False
            for point in (pts[0], pts[-1]):
                key = (round(point[0], 6), round(point[1], 6))
                endpoints[key] = endpoints.get(key, 0) + 1
        return bool(endpoints) and all(count == 2 for count in endpoints.values())


_DESCRIPTOR_SIZE = (
    len(_ENTITY_KINDS)
    + len(_CONSTRAINT_KINDS)
    + len(_DIMENSION_KINDS)
    + 8  # construction ratio, closed flag, bounds, dimensions count, ...
)


def sketch_document_descriptor(document: SketchDocument) -> torch.Tensor:
    """Deterministic structural descriptor of a sketch."""
    vec = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    offset = 0

    construction = 0
    total = 0
    for entity in document.entities:
        total += 1
        if entity.is_construction:
            construction += 1
        vec[offset + _ENTITY_KINDS.index(entity.kind)] += 1.0
    vec[offset : offset + len(_ENTITY_KINDS)] /= max(total, 1)
    offset += len(_ENTITY_KINDS)

    for constraint in document.constraints:
        lowered = constraint.lower()
        for i, kind in enumerate(_CONSTRAINT_KINDS):
            if kind in lowered:
                vec[offset + i] += 1.0
    vec[offset : offset + len(_CONSTRAINT_KINDS)] /= max(len(document.constraints), 1)
    offset += len(_CONSTRAINT_KINDS)

    dimension_values: list[float] = []
    for dimension in document.dimensions:
        kind = str(dimension.get("kind", "linear"))
        if kind in _DIMENSION_KINDS:
            vec[offset + _DIMENSION_KINDS.index(kind)] += 1.0
        with contextlib.suppress(TypeError, ValueError):
            dimension_values.append(abs(float(dimension.get("value", 0.0))))
    vec[offset : offset + len(_DIMENSION_KINDS)] /= max(len(document.dimensions), 1)
    offset += len(_DIMENSION_KINDS)

    vec[offset + 0] = math.log1p(total)
    vec[offset + 1] = construction / max(total, 1)
    vec[offset + 2] = 1.0 if document.is_closed else 0.0
    vec[offset + 3] = 1.0 if document.source == "hand" else 0.0
    if document.bounds is not None:
        (x0, y0), (x1, y1) = document.bounds
        vec[offset + 4] = max(x1 - x0, 0.0)
        vec[offset + 5] = max(y1 - y0, 0.0)
        vec[offset + 6] = x1 - x0 if x1 - x0 > 0 else 0.0
    vec[offset + 7] = sum(dimension_values) if dimension_values else 0.0
    return vec


class SketchEncoder(MultimodalEncoder):
    """Encoder for the ``sketch`` modality."""

    modality: ClassVar[Modality] = Modality.SKETCH

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
        if x.dim() != 2 or x.shape[-1] != _DESCRIPTOR_SIZE:
            raise ValueError(
                f"sketch encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, SketchDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty sketch batch")
        descriptors = torch.stack([sketch_document_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = [
    "SketchDocument",
    "SketchEncoder",
    "SketchEntity",
    "sketch_document_descriptor",
]
