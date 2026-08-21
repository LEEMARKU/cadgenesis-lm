"""cadgenesis.world_model.objects
================================
Internal world-model object representation (Pillar 4).

The world model's central abstraction is the :class:`WorldObject` — a
normalised, reasoning-friendly record of every entity the model reasons
about: CAD features, materials, poses, boundary conditions, simulation
results and the object graph linking them.

Design rules
------------
* Reuses ``cadgenesis.cad.geometry.core`` for all vector / transform math
  (never re-implements it).
* Objects are plain data + a small pose API; behaviour lives in the
  reasoners (``spatial.py``, ``mechanical.py``, ...).
* Everything serializes to/from plain dicts for the memory system and for
  plan/audit traces.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Transform, Vec

# Canonical primitive feature families the world model knows about.
PRIMITIVE_FAMILIES: tuple[str, ...] = (
    "block",
    "cylinder",
    "sphere",
    "cone",
    "torus",
    "prism",
    "hole",
    "fillet",
    "chamfer",
    "extrusion",
    "revolve",
    "loft",
)


_ID_COUNTER = itertools.count(1)


def _next_id() -> str:
    return f"wo-{next(_ID_COUNTER)}"


@dataclass
class Material:
    """Physical material properties (SI-ish engineering units)."""

    name: str = "steel"
    density_kg_m3: float = 7850.0
    yield_strength_mpa: float = 250.0
    elastic_modulus_gpa: float = 210.0
    poisson_ratio: float = 0.30
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "density_kg_m3": self.density_kg_m3,
            "yield_strength_mpa": self.yield_strength_mpa,
            "elastic_modulus_gpa": self.elastic_modulus_gpa,
            "poisson_ratio": self.poisson_ratio,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Material:
        return cls(
            name=data.get("name", "steel"),
            density_kg_m3=float(data.get("density_kg_m3", 7850.0)),
            yield_strength_mpa=float(data.get("yield_strength_mpa", 250.0)),
            elastic_modulus_gpa=float(data.get("elastic_modulus_gpa", 210.0)),
            poisson_ratio=float(data.get("poisson_ratio", 0.30)),
            properties=dict(data.get("properties", {})),
        )


# A few stock materials used by tests, benchmarks and the simulator.
STOCK_MATERIALS: dict[str, Material] = {
    "steel": Material("steel", 7850.0, 250.0, 210.0, 0.30, {"cost_per_kg": 1.5}),
    "aluminum": Material("aluminum", 2700.0, 95.0, 69.0, 0.33, {"cost_per_kg": 3.0}),
    "titanium": Material("titanium", 4510.0, 880.0, 116.0, 0.34, {"cost_per_kg": 30.0}),
    "plastic": Material("plastic", 1400.0, 40.0, 3.0, 0.40, {"cost_per_kg": 2.0}),
}


@dataclass
class BoundaryCondition:
    """A force / torque / fixation applied to an object."""

    kind: str  # "force" | "torque" | "fixation" | "pressure"
    magnitude: float = 0.0
    axis: Vec = field(default_factory=lambda: Vec(0.0, 0.0, 1.0))
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "magnitude": self.magnitude,
            "axis": self.axis.to_tuple(),
            "name": self.name,
        }


@dataclass
class LoadCase:
    """A named set of boundary conditions acting on the design."""

    name: str
    conditions: list[BoundaryCondition] = field(default_factory=list)
    safety_factor_target: float = 2.0

    def add(self, kind: str, magnitude: float, axis: Vec) -> BoundaryCondition:
        condition = BoundaryCondition(kind=kind, magnitude=magnitude, axis=axis)
        self.conditions.append(condition)
        return condition


@dataclass
class WorldObject:
    """Normalised world-model object.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "mounting bracket").
    feature : str
        Feature family (one of :data:`PRIMITIVE_FAMILIES` or a CAD feature).
    parameters : dict[str, Any]
        Feature parameters (dimensions, angles, radii, ...).
    material : Material | None
        Optional physical material.
    pose : Transform | None
        World pose; defaults to identity.
    confidence : float
        Model confidence in this object (0..1).
    parent : str | None
        Parent object id (assembly tree).
    """

    name: str = "object"
    feature: str = "block"
    parameters: dict[str, Any] = field(default_factory=dict)
    material: Material | None = None
    pose: Transform | None = None
    confidence: float = 1.0
    parent: str | None = None
    object_id: str = field(default_factory=_next_id)
    state: dict[str, Any] = field(default_factory=dict)
    relations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feature not in PRIMITIVE_FAMILIES:
            raise ValueError(
                f"unknown feature family {self.feature!r}; expected one of {PRIMITIVE_FAMILIES}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.pose is None:
            self.pose = Transform.identity()

    # ------------------------------------------------------------- geometry

    @property
    def position(self) -> Vec:
        assert self.pose is not None
        return self.pose.apply(Vec(0.0, 0.0, 0.0))

    def bounds(self) -> tuple[Vec, Vec]:
        """Axis-aligned bounding box ``(min, max)`` from the feature family.

        ``parameters`` describe full dimensions, so half-extents are derived
        by halving them (cylinder/sphere radii are already half-extents).
        """
        feature = self.feature
        params = self.parameters
        if feature == "block":
            x = float(params.get("length", params.get("x", 10.0))) / 2.0
            y = float(params.get("width", params.get("y", 10.0))) / 2.0
            z = float(params.get("height", params.get("z", 10.0))) / 2.0
        elif feature in ("cylinder", "revolve"):
            r = float(params.get("radius", 5.0))
            h = float(params.get("height", float(params.get("length", 10.0)))) / 2.0
            x, y, z = r, r, h
        elif feature == "sphere":
            r = float(params.get("radius", 5.0))
            x, y, z = r, r, r
        elif feature == "cone":
            r = float(params.get("radius", 5.0))
            h = float(params.get("height", 10.0)) / 2.0
            x, y, z = r, r, h
        elif feature == "torus":
            r = float(params.get("radius", 5.0)) + float(params.get("tube", 1.0))
            x, y, z = r, r, r
        elif feature == "hole":
            r = float(params.get("radius", 2.0))
            d = float(params.get("depth", 5.0)) / 2.0
            x, y, z = r, r, d
        else:
            # Best-effort half-extents from numeric parameters.
            values = [abs(float(v)) for v in params.values() if isinstance(v, (int, float))]
            x = max(values[:1] or [10.0])
            y = max(values[1:2] or [10.0])
            z = max(values[2:3] or [10.0])
        center = self.position
        assert self.pose is not None
        return (
            Vec(center.x - x, center.y - y, center.z - z),
            Vec(center.x + x, center.y + y, center.z + z),
        )

    def volume_estimate(self) -> float:
        """Analytic volume of the primitive (world frame ~ pose-agnostic)."""
        feature = self.feature
        p = self.parameters
        try:
            if feature == "block":
                return (
                    float(p.get("length", 10.0))
                    * float(p.get("width", 10.0))
                    * float(p.get("height", 10.0))
                )
            if feature in ("cylinder", "revolve"):
                import math

                r = float(p.get("radius", 5.0))
                h = float(p.get("height", float(p.get("length", 10.0))))
                return math.pi * r * r * h
            if feature == "sphere":
                import math

                return 4.0 / 3.0 * math.pi * float(p.get("radius", 5.0)) ** 3
            if feature == "cone":
                import math

                r = float(p.get("radius", 5.0))
                h = float(p.get("height", 10.0))
                return math.pi * r * r * h / 3.0
            if feature == "hole":
                import math

                r = float(p.get("radius", 2.0))
                d = float(p.get("depth", 5.0))
                return math.pi * r * r * d
        except (TypeError, ValueError):
            pass
        return 0.0

    def mass(self) -> float:
        """Mass estimate in kg (0 when no material is set)."""
        if self.material is None:
            return 0.0
        return self.volume_estimate() * self.material.density_kg_m3 / 1e9  # mm^3 -> m^3

    # ---------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        assert self.pose is not None
        return {
            "name": self.name,
            "feature": self.feature,
            "parameters": dict(self.parameters),
            "material": self.material.to_dict() if self.material else None,
            "pose": self.pose.to_list(),
            "confidence": self.confidence,
            "parent": self.parent,
            "object_id": self.object_id,
            "state": dict(self.state),
            "relations": dict(self.relations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldObject:
        material = Material.from_dict(data["material"]) if data.get("material") else None
        pose_data = data.get("pose")
        pose = Transform(pose_data) if isinstance(pose_data, list) and len(pose_data) == 4 else None
        return cls(
            name=data.get("name", "object"),
            feature=data.get("feature", "block"),
            parameters=dict(data.get("parameters", {})),
            material=material,
            pose=pose,
            confidence=float(data.get("confidence", 1.0)),
            parent=data.get("parent"),
            object_id=data.get("object_id", _next_id()),
            state=dict(data.get("state", {})),
            relations=dict(data.get("relations", {})),
        )


def make_object(
    feature: str,
    name: str,
    parameters: dict[str, Any] | None = None,
    material: str | Material | None = None,
    **kwargs: Any,
) -> WorldObject:
    """Convenience factory for a :class:`WorldObject`."""
    resolved_material: Material | None = None
    if isinstance(material, str):
        resolved_material = STOCK_MATERIALS.get(
            material.lower().replace(" ", "_"), Material(name=material)
        )
    elif isinstance(material, Material):
        resolved_material = material
    return WorldObject(
        name=name,
        feature=feature,
        parameters=dict(parameters or {}),
        material=resolved_material,
        **kwargs,
    )


@dataclass
class ObjectGraph:
    """The world-model object graph: objects + parent/child links."""

    objects: list[WorldObject] = field(default_factory=list)

    def add(self, obj: WorldObject) -> WorldObject:
        self.objects.append(obj)
        return obj

    def get(self, object_id: str) -> WorldObject | None:
        return next((o for o in self.objects if o.object_id == object_id), None)

    def children(self, object_id: str) -> list[WorldObject]:
        return [o for o in self.objects if o.parent == object_id]

    def roots(self) -> list[WorldObject]:
        return [o for o in self.objects if o.parent is None]

    def neighbors(self, object_id: str) -> list[str]:
        """Ids of directly related objects (parent, children, relations)."""
        result: list[str] = []
        for obj in self.objects:
            if obj.object_id == object_id:
                continue
            related = obj.parent == object_id or obj.object_id in set(
                self.relations_of(object_id).get("children", [])
            )
            if related:
                result.append(obj.object_id)
            if object_id in set(self.relations_of(obj.object_id).get("children", [])):
                result.append(obj.object_id)
        return result

    def relations_of(self, object_id: str) -> dict[str, Any]:
        obj = self.get(object_id)
        return dict(obj.relations) if obj else {}

    def relate(
        self,
        parent: str,
        child: str,
        relation: str = "mounts",
    ) -> None:
        """Link ``child`` under ``parent`` in the assembly tree."""
        parent_obj = self.get(parent)
        child_obj = self.get(child)
        if parent_obj is None or child_obj is None:
            raise KeyError(f"unknown object in relate: {parent!r} or {child!r}")
        if child_obj.parent is not None and child_obj.parent != parent:
            raise ValueError(f"{child} already has a parent {child_obj.parent!r}")
        child_obj.parent = parent
        parent_obj.relations.setdefault("children", [])
        if child not in parent_obj.relations["children"]:
            parent_obj.relations["children"].append(child)

    def set_pose(self, object_id: str, pose: Transform | dict[str, Any]) -> None:
        """Set an object's world pose from a Transform or 4x4 matrix dict."""
        obj = self.get(object_id)
        if obj is None:
            raise KeyError(f"unknown object {object_id!r}")
        if isinstance(pose, Transform):
            obj.pose = pose
        elif isinstance(pose, dict):
            matrix = pose.get("matrix") or pose.get("pose")
            if not isinstance(matrix, list) or len(matrix) != 4:
                raise ValueError("pose dict requires a 4x4 'matrix' list")
            obj.pose = Transform(matrix)
        else:
            raise TypeError("pose must be a Transform or a 4x4 matrix dict")

    def root_for(self, obj: WorldObject) -> WorldObject | None:
        """The topmost ancestor of an object (itself when it is a root)."""
        current: WorldObject | None = obj
        seen: set[str] = set()
        while current is not None and current.parent is not None:
            if current.object_id in seen:
                return current
            seen.add(current.object_id)
            current = self.get(current.parent)
        return current

    def __len__(self) -> int:
        return len(self.objects)

    def to_dict(self) -> dict[str, Any]:
        return {"objects": [o.to_dict() for o in self.objects]}


__all__ = [
    "PRIMITIVE_FAMILIES",
    "STOCK_MATERIALS",
    "BoundaryCondition",
    "LoadCase",
    "Material",
    "ObjectGraph",
    "WorldObject",
    "make_object",
]
