"""cadgenesis.cad.modeling.primitives
==================================
Analytic solid primitives used as CSG leaves and B-Rep sources: box,
cylinder, sphere, cone and torus, each with volume, surface area and
axis-aligned bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Vec

_PRIMITIVE_KINDS = ("box", "cylinder", "sphere", "cone", "torus")


@dataclass
class SolidPrimitive:
    """A parametric analytic solid primitive."""

    kind: str
    dims: dict[str, float] = field(default_factory=dict)
    position: Vec = field(default_factory=lambda: Vec(0, 0, 0))

    def __post_init__(self) -> None:
        if self.kind not in _PRIMITIVE_KINDS:
            raise ValueError(
                f"unsupported primitive kind {self.kind!r}; expected one of {_PRIMITIVE_KINDS}"
            )
        if not isinstance(self.position, Vec):
            self.position = Vec.from_sequence(self.position)
        required = self.required_dimensions()
        missing = [d for d in required if d not in self.dims]
        if missing:
            raise ValueError(f"primitive {self.kind!r} missing dimensions: {missing}")

    @property
    def name(self) -> str:
        return f"{self.kind}_solid"

    def required_dimensions(self) -> tuple[str, ...]:
        return {
            "box": ("length", "width", "height"),
            "cylinder": ("radius", "height"),
            "sphere": ("radius",),
            "cone": ("radius", "height"),
            "torus": ("major_radius", "minor_radius"),
        }[self.kind]

    def volume(self) -> float:
        d = self.dims
        if self.kind == "box":
            return float(d["length"] * d["width"] * d["height"])
        if self.kind == "cylinder":
            return math.pi * d["radius"] ** 2 * d["height"]
        if self.kind == "sphere":
            return (4.0 / 3.0) * math.pi * d["radius"] ** 3
        if self.kind == "cone":
            return (1.0 / 3.0) * math.pi * d["radius"] ** 2 * d["height"]
        return 2.0 * math.pi**2 * d["major_radius"] * d["minor_radius"] ** 2

    def surface_area(self) -> float:
        d = self.dims
        if self.kind == "box":
            return 2.0 * (
                d["length"] * d["width"] + d["width"] * d["height"] + d["length"] * d["height"]
            )
        if self.kind == "cylinder":
            return 2.0 * math.pi * d["radius"] * (d["radius"] + d["height"])
        if self.kind == "sphere":
            return 4.0 * math.pi * d["radius"] ** 2
        if self.kind == "cone":
            slant = math.sqrt(d["radius"] ** 2 + d["height"] ** 2)
            return math.pi * d["radius"] * (d["radius"] + slant)
        return 4.0 * math.pi**2 * d["major_radius"] * d["minor_radius"]

    def aabb(self) -> tuple[Vec, Vec]:
        """Axis-aligned bounds as (min, max) with the primitive centred on ``position``."""
        d = self.dims
        if self.kind == "box":
            half = Vec(d["length"] / 2, d["width"] / 2, d["height"] / 2)
        elif self.kind in ("cylinder", "cone"):
            half = Vec(d["radius"], d["radius"], d["height"] / 2)
        elif self.kind == "sphere":
            half = Vec(d["radius"], d["radius"], d["radius"])
        else:  # torus
            extent = d["major_radius"] + d["minor_radius"]
            half = Vec(extent, extent, d["minor_radius"])
        return self.position - half, self.position + half

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dims": dict(self.dims),
            "position": self.position.to_list(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolidPrimitive:
        return cls(
            kind=str(data["kind"]),
            dims={str(k): float(v) for k, v in data.get("dims", {}).items()},
            position=Vec(*[float(v) for v in data.get("position", [0, 0, 0])]),
        )


def make_box(
    length: float, width: float, height: float, position: Vec | None = None, name: str = ""
) -> SolidPrimitive:
    return SolidPrimitive(
        "box", {"length": length, "width": width, "height": height}, position or Vec(0, 0, 0)
    )


def make_cylinder(radius: float, height: float, position: Vec | None = None) -> SolidPrimitive:
    return SolidPrimitive(
        "cylinder", {"radius": radius, "height": height}, position or Vec(0, 0, 0)
    )


def make_sphere(radius: float, position: Vec | None = None) -> SolidPrimitive:
    return SolidPrimitive("sphere", {"radius": radius}, position or Vec(0, 0, 0))


def make_cone(radius: float, height: float, position: Vec | None = None) -> SolidPrimitive:
    return SolidPrimitive("cone", {"radius": radius, "height": height}, position or Vec(0, 0, 0))


def make_torus(
    major_radius: float, minor_radius: float, position: Vec | None = None
) -> SolidPrimitive:
    return SolidPrimitive(
        "torus",
        {"major_radius": major_radius, "minor_radius": minor_radius},
        position or Vec(0, 0, 0),
    )


__all__ = [
    "SolidPrimitive",
    "make_box",
    "make_cone",
    "make_cylinder",
    "make_sphere",
    "make_torus",
]
