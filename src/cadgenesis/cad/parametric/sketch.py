"""cadgenesis.cad.parametric.sketch
=================================
Sketch modelling: 2D sketch entities (points, lines, arcs, circles, splines,
dimensions) and the :class:`Sketch` document that holds them together with
their geometric constraints on a sketch plane.

A :class:`SketchProfile` is a closed or open sequence of entities used as the
input to a feature (extrude, revolve, sweep, loft...).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from cadgenesis.cad.geometry.core import Plane, Vec

EntityT = TypeVar("EntityT", bound="SketchEntity")

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class SketchEntity:
    """Base class for a 2D sketch entity (coordinates in sketch space)."""

    name: str = ""
    is_construction: bool = False
    kind = "entity"

    def points(self) -> list[Vec]:
        """The defining 2D points of the entity (sketch plane coords)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "construction": self.is_construction}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SketchEntity:
        kind = data["kind"]
        name = str(data.get("name", ""))
        construction = bool(data.get("construction", False))
        if kind == "point":
            p = data["point"]
            return PointEntity(name=name, point=Vec(p[0], p[1]), is_construction=construction)
        if kind == "line":
            a, b = data["start"], data["end"]
            return LineEntity(
                name=name, start=Vec(a[0], a[1]), end=Vec(b[0], b[1]), is_construction=construction
            )
        if kind == "circle":
            c, r = data["center"], data["radius"]
            return CircleEntity(
                name=name, center=Vec(c[0], c[1]), radius=float(r), is_construction=construction
            )
        if kind == "arc":
            c, r = data["center"], data["radius"]
            return ArcEntity(
                name=name,
                center=Vec(c[0], c[1]),
                radius=float(r),
                start_angle=float(data["start_angle"]),
                end_angle=float(data["end_angle"]),
                is_construction=construction,
            )
        if kind == "spline":
            pts = [Vec(p[0], p[1]) for p in data["control_points"]]
            return SplineEntity(name=name, control_points=pts, is_construction=construction)
        raise ValueError(f"unknown sketch entity kind {kind!r}")


@dataclass
class PointEntity(SketchEntity):
    point: Vec = field(default_factory=lambda: Vec(0, 0))
    kind = "point"

    def __post_init__(self) -> None:
        if not isinstance(self.point, Vec):
            self.point = Vec.from_sequence(self.point)

    def points(self) -> list[Vec]:
        return [self.point]


@dataclass
class LineEntity(SketchEntity):
    start: Vec = field(default_factory=lambda: Vec(0, 0))
    end: Vec = field(default_factory=lambda: Vec(1, 0))
    kind = "line"

    def __post_init__(self) -> None:
        if not isinstance(self.start, Vec):
            self.start = Vec.from_sequence(self.start)
        if not isinstance(self.end, Vec):
            self.end = Vec.from_sequence(self.end)

    def points(self) -> list[Vec]:
        return [self.start, self.end]

    def direction(self) -> Vec:
        return self.end - self.start

    def length(self) -> float:
        return (self.end - self.start).norm()

    def midpoint(self) -> Vec:
        return self.start + (self.end - self.start) * 0.5


@dataclass
class CircleEntity(SketchEntity):
    center: Vec = field(default_factory=lambda: Vec(0, 0))
    radius: float = 1.0
    kind = "circle"

    def __post_init__(self) -> None:
        if not isinstance(self.center, Vec):
            self.center = Vec.from_sequence(self.center)
        if self.radius <= 0:
            raise ValueError("circle radius must be positive")

    def points(self) -> list[Vec]:
        return [self.center]


@dataclass
class ArcEntity(SketchEntity):
    center: Vec = field(default_factory=lambda: Vec(0, 0))
    radius: float = 1.0
    start_angle: float = 0.0
    end_angle: float = math.pi
    kind = "arc"

    def __post_init__(self) -> None:
        if not isinstance(self.center, Vec):
            self.center = Vec.from_sequence(self.center)
        if self.radius <= 0:
            raise ValueError("arc radius must be positive")

    def points(self) -> list[Vec]:
        return [self.center]

    def start_point(self) -> Vec:
        return Vec(
            self.center.x + self.radius * math.cos(self.start_angle),
            self.center.y + self.radius * math.sin(self.start_angle),
        )

    def end_point(self) -> Vec:
        return Vec(
            self.center.x + self.radius * math.cos(self.end_angle),
            self.center.y + self.radius * math.sin(self.end_angle),
        )

    def sweep_angle(self) -> float:
        sweep = (self.end_angle - self.start_angle) % (2 * math.pi)
        return sweep if sweep != 0.0 else 2 * math.pi


@dataclass
class SplineEntity(SketchEntity):
    control_points: list[Vec] = field(default_factory=list)
    kind = "spline"

    def __post_init__(self) -> None:
        self.control_points = [Vec.from_sequence(p) for p in self.control_points]
        if len(self.control_points) < 2:
            raise ValueError("a spline needs at least 2 control points")

    def points(self) -> list[Vec]:
        return self.control_points


# ---------------------------------------------------------------------------
# Sketch document
# ---------------------------------------------------------------------------


@dataclass
class Dimension:
    """A dimensional constraint: distance / angle / radius with a value."""

    kind: str  # "linear" | "angular" | "radius" | "diameter"
    value: float
    entity_a: str = ""
    entity_b: str = ""
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dimension:
        return cls(
            kind=str(data["kind"]),
            value=float(data["value"]),
            entity_a=str(data.get("entity_a", "")),
            entity_b=str(data.get("entity_b", "")),
            name=str(data.get("name", "")),
        )


class Sketch:
    """A 2D parametric sketch living on a sketch plane.

    Holds entities by name, dimensional constraints (:class:`Dimension`) and
    geometric constraints (imported lazily from :mod:`constraints` to avoid a
    circular import).
    """

    def __init__(self, name: str = "sketch", plane: Plane | None = None) -> None:
        self.name = name
        self.plane = plane if plane is not None else Plane.xy()
        self._entities: dict[str, SketchEntity] = {}
        self.dimensions: list[Dimension] = []
        self._constraints: list[Any] = []
        self._unnamed = 0

    # -- entities ------------------------------------------------------------
    def add_entity(self, entity: EntityT) -> EntityT:
        if entity.name and entity.name in self._entities:
            raise KeyError(f"sketch entity {entity.name!r} already exists")
        if not entity.name:
            self._unnamed += 1
            entity.name = f"{entity.kind}_{self._unnamed}"
        self._entities[entity.name] = entity
        return entity

    def add_point(
        self, x: float, y: float, name: str = "", construction: bool = False
    ) -> PointEntity:
        return self.add_entity(
            PointEntity(name=name, point=Vec(x, y), is_construction=construction)
        )

    def add_line(
        self, start: Vec, end: Vec, name: str = "", construction: bool = False
    ) -> LineEntity:
        return self.add_entity(
            LineEntity(name=name, start=start, end=end, is_construction=construction)
        )

    def add_circle(
        self, center: Vec, radius: float, name: str = "", construction: bool = False
    ) -> CircleEntity:
        return self.add_entity(
            CircleEntity(name=name, center=center, radius=radius, is_construction=construction)
        )

    def add_arc(
        self,
        center: Vec,
        radius: float,
        start_angle: float,
        end_angle: float,
        name: str = "",
        construction: bool = False,
    ) -> ArcEntity:
        return self.add_entity(
            ArcEntity(
                name=name,
                center=center,
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                is_construction=construction,
            )
        )

    def add_spline(
        self, control_points: list[Vec], name: str = "", construction: bool = False
    ) -> SplineEntity:
        return self.add_entity(
            SplineEntity(name=name, control_points=control_points, is_construction=construction)
        )

    def entity(self, name: str) -> SketchEntity:
        if name not in self._entities:
            raise KeyError(f"unknown sketch entity {name!r}")
        return self._entities[name]

    @property
    def entities(self) -> list[SketchEntity]:
        return list(self._entities.values())

    def entity_names(self) -> list[str]:
        return list(self._entities)

    # -- constraints -----------------------------------------------------------
    def add_constraint(self, constraint: Any) -> Any:
        """Add a geometric constraint (see :mod:`cadgenesis.cad.parametric.constraints`)."""
        if constraint.name and any(c.name == constraint.name for c in self._constraints):
            raise KeyError(f"constraint {constraint.name!r} already exists")
        self._constraints.append(constraint)
        return constraint

    def add_dimension(self, dimension: Dimension) -> Dimension:
        self.dimensions.append(dimension)
        return dimension

    @property
    def constraints(self) -> list[Any]:
        return list(self._constraints)

    # -- convenience ----------------------------------------------------------
    def rectangle(
        self,
        x0: float,
        y0: float,
        width: float,
        height: float,
        name: str = "rect",
        construction: bool = False,
    ) -> None:
        """Add an axis-aligned rectangle as four construction-aware lines."""
        lines = [
            (Vec(x0, y0), Vec(x0 + width, y0)),
            (Vec(x0 + width, y0), Vec(x0 + width, y0 + height)),
            (Vec(x0 + width, y0 + height), Vec(x0, y0 + height)),
            (Vec(x0, y0 + height), Vec(x0, y0)),
        ]
        for i, (a, b) in enumerate(lines):
            self.add_line(a, b, name=f"{name}_e{i}", construction=construction)

    def is_closed_profile(self) -> bool:
        """True when the non-construction entities form a closed loop."""
        endpoints: dict[Vec, int] = {}
        for entity in self.entities:
            if entity.is_construction:
                continue
            pts = entity.points()
            if len(pts) == 2:
                for p in pts:
                    key = Vec(round(p.x, 6), round(p.y, 6))
                    endpoints[key] = endpoints.get(key, 0) + 1
            elif hasattr(entity, "start_point") and hasattr(entity, "end_point"):
                for p in (entity.start_point(), entity.end_point()):
                    key = Vec(round(p.x, 6), round(p.y, 6))
                    endpoints[key] = endpoints.get(key, 0) + 1
        return bool(endpoints) and all(count == 2 for count in endpoints.values())

    def bounds(self) -> tuple[Vec, Vec]:
        """Axis-aligned 2D bounds as (min, max)."""
        xs: list[float] = []
        ys: list[float] = []
        for entity in self.entities:
            for p in entity.points():
                xs.append(p.x)
                ys.append(p.y)
            if isinstance(entity, ArcEntity):
                xs.extend((entity.start_point().x, entity.end_point().x))
                ys.extend((entity.start_point().y, entity.end_point().y))
        if not xs:
            return Vec(0, 0), Vec(0, 0)
        return Vec(min(xs), min(ys)), Vec(max(xs), max(ys))

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plane": {
                "point": self.plane.point.to_list(),
                "normal": self.plane.normal.to_list(),
            },
            "entities": [e.to_dict() for e in self.entities],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "constraints": [c.to_dict() for c in self._constraints],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sketch:
        plane_data = data.get("plane", {})
        plane = Plane.xy()
        if plane_data:
            plane = Plane(
                Vec(*[float(v) for v in plane_data["point"]]),
                Vec(*[float(v) for v in plane_data["normal"]]),
            )
        sketch = cls(name=str(data.get("name", "sketch")), plane=plane)
        for entity_data in data.get("entities", []):
            sketch.add_entity(SketchEntity.from_dict(entity_data))
        for dim_data in data.get("dimensions", []):
            sketch.add_dimension(Dimension.from_dict(dim_data))
        from cadgenesis.cad.parametric.constraints import GeometricConstraint

        for c_data in data.get("constraints", []):
            sketch.add_constraint(GeometricConstraint.from_dict(c_data))
        return sketch


# ---------------------------------------------------------------------------
# Sketch profile
# ---------------------------------------------------------------------------


class SketchProfile:
    """A boundary loop extracted from a :class:`Sketch`.

    ``entities`` is the ordered list of sketch entity names forming the
    loop; ``is_closed`` marks whether the loop is closed (extrude-able to a
    solid) or open (used for thin/rib/surface features).
    """

    def __init__(self, sketch: Sketch, entities: Sequence[str], is_closed: bool = True) -> None:
        self.sketch = sketch
        self.entity_names = list(entities)
        self.is_closed = is_closed
        if not self.entity_names:
            raise ValueError("a sketch profile needs at least one entity")

    def entities(self) -> list[SketchEntity]:
        return [self.sketch.entity(name) for name in self.entity_names]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sketch": self.sketch.name,
            "entities": self.entity_names,
            "is_closed": self.is_closed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], sketch: Sketch) -> SketchProfile:
        return cls(sketch, data["entities"], bool(data.get("is_closed", True)))


__all__ = [
    "ArcEntity",
    "CircleEntity",
    "Dimension",
    "LineEntity",
    "PointEntity",
    "Sketch",
    "SketchEntity",
    "SketchProfile",
    "SplineEntity",
]
