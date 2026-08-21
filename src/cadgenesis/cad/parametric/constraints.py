"""cadgenesis.cad.parametric.constraints
======================================
Geometric constraints for sketch modelling and their numeric solver.

Supported geometric constraint types (the core CAD constraint vocabulary):
parallel, perpendicular, tangent, concentric, coincident, equal, symmetry,
plus horizontal, vertical, collinear, midpoint, fixed and on-axis.

The :class:`SketchConstraintSolver` performs:
1. **DOF analysis** — counts degrees of freedom consumed by constraints and
   classifies the sketch as *under / fully / over* constrained.
2. **Numeric resolution** — a projection-based iterative solver for
   point-based constraints (coincident, horizontal, vertical, concentric,
   midpoint, symmetric, fixed, equal-length, distance dimensions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Vec
from cadgenesis.cad.parametric.sketch import (
    ArcEntity,
    CircleEntity,
    LineEntity,
    PointEntity,
    Sketch,
    SketchEntity,
)

CONSTRAINT_TYPES = (
    "COINCIDENT",
    "COLLINEAR",
    "COPLANAR",
    "PARALLEL",
    "PERPENDICULAR",
    "HORIZONTAL",
    "VERTICAL",
    "TANGENT",
    "CURVATURE",
    "SYMMETRIC",
    "MIDPOINT",
    "EQUAL",
    "EQUAL_LENGTH",
    "EQUAL_RADIUS",
    "FIXED",
    "CONCENTRIC",
    "ON_CURVE",
)

# number of DOF consumed per constraint type
_CONSTRAINT_DOF: dict[str, int] = {
    "COINCIDENT": 2,
    "COLLINEAR": 2,
    "COPLANAR": 1,
    "PARALLEL": 1,
    "PERPENDICULAR": 1,
    "HORIZONTAL": 1,
    "VERTICAL": 1,
    "TANGENT": 1,
    "CURVATURE": 1,
    "SYMMETRIC": 2,
    "MIDPOINT": 2,
    "EQUAL": 2,
    "EQUAL_LENGTH": 1,
    "EQUAL_RADIUS": 1,
    "FIXED": 2,
    "CONCENTRIC": 2,
    "ON_CURVE": 1,
}

# degrees of freedom contributed by each entity kind (2D)
_ENTITY_DOF: dict[str, int] = {
    "point": 2,
    "line": 4,
    "circle": 3,
    "arc": 5,
    "spline": 2 * 4,  # each control point
}


class GeometricConstraint:
    """A geometric relationship between two sketch entities."""

    def __init__(
        self,
        constraint_type: str,
        entity_a: str,
        entity_b: str = "",
        name: str = "",
        tolerance: float = 1e-6,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if constraint_type not in CONSTRAINT_TYPES:
            raise ValueError(
                f"unknown constraint type {constraint_type!r}; expected one of {CONSTRAINT_TYPES}"
            )
        if not entity_a:
            raise ValueError("constraint needs at least one entity")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.constraint_type = constraint_type
        self.entity_a = entity_a
        self.entity_b = entity_b
        self.name = name or f"{constraint_type.lower()}_{entity_a}"
        self.tolerance = tolerance
        self.metadata = dict(metadata or {})

    @property
    def consumes_dof(self) -> int:
        return _CONSTRAINT_DOF[self.constraint_type]

    def references(self, entity_name: str) -> bool:
        return entity_name in (self.entity_a, self.entity_b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "name": self.name,
            "tolerance": self.tolerance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeometricConstraint:
        return cls(
            constraint_type=str(data["type"]),
            entity_a=str(data["entity_a"]),
            entity_b=str(data.get("entity_b", "")),
            name=str(data.get("name", "")),
            tolerance=float(data.get("tolerance", 1e-6)),
            metadata=data.get("metadata"),
        )

    def __repr__(self) -> str:
        return f"GeometricConstraint({self.constraint_type}, {self.entity_a!r}, {self.entity_b!r})"


@dataclass
class ConstraintSolution:
    """Result of solving a sketch's constraint system."""

    status: str  # "under" | "fully" | "over"
    dof: int
    residual: float
    iterations: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def is_fully_constrained(self) -> bool:
        return self.status == "fully"

    @property
    def is_over_constrained(self) -> bool:
        return self.status == "over"


def _point_names(sketch: Sketch) -> list[str]:
    """Names of entities that participate in point-based solving."""
    return [
        entity.name
        for entity in sketch.entities
        if isinstance(entity, (PointEntity, LineEntity, CircleEntity, ArcEntity))
    ]


def _point_value(sketch: Sketch, name: str) -> Vec | None:
    entity = sketch.entity(name)
    if isinstance(entity, PointEntity):
        return Vec(entity.point.x, entity.point.y)
    if isinstance(entity, LineEntity):
        return Vec(entity.start.x, entity.start.y)
    if isinstance(entity, (CircleEntity, ArcEntity)):
        return Vec(entity.center.x, entity.center.y)
    return None


def _set_point(sketch: Sketch, name: str, value: Vec) -> None:
    entity = sketch.entity(name)
    if isinstance(entity, PointEntity):
        entity.point = Vec(value.x, value.y)
    elif isinstance(entity, LineEntity):
        entity.start = Vec(value.x, value.y)
    elif isinstance(entity, (CircleEntity, ArcEntity)):
        entity.center = Vec(value.x, value.y)


def _entity_residual(
    sketch: Sketch, constraint: GeometricConstraint
) -> tuple[float, list[tuple[str, Vec]]]:
    """Compute the residual and per-entity gradient corrections for a constraint."""
    t = constraint.constraint_type
    a = constraint.entity_a
    b = constraint.entity_b
    eps = 1e-6
    if t == "FIXED":
        return 0.0, []
    if t == "COINCIDENT":
        pa = _point_value(sketch, a)
        pb = _point_value(sketch, b)
        if pa is None or pb is None:
            return 0.0, []
        delta = pa - pb
        residual = delta.norm()
        return residual, [(a, -delta * 0.5), (b, delta * 0.5)]
    if t == "CONCENTRIC":
        pa = _point_value(sketch, a)
        pb = _point_value(sketch, b)
        if pa is None or pb is None:
            return 0.0, []
        delta = pa - pb
        return delta.norm(), [(a, -delta * 0.5), (b, delta * 0.5)]
    if t == "MIDPOINT":
        line = sketch.entity(b)
        point = a
        if not isinstance(line, LineEntity):
            return 0.0, []
        pa = _point_value(sketch, point)
        if pa is None:
            return 0.0, []
        mid = line.midpoint()
        delta = pa - mid
        return delta.norm(), [(point, -delta), (b, delta * 0.5)]
    if t == "HORIZONTAL":
        entity = sketch.entity(a)
        if isinstance(entity, LineEntity):
            return abs(entity.start.y - entity.end.y), [
                (a, Vec(0, (entity.end.y - entity.start.y)))
            ]
        return 0.0, []
    if t == "VERTICAL":
        entity = sketch.entity(a)
        if isinstance(entity, LineEntity):
            return abs(entity.start.x - entity.end.x), [
                (a, Vec((entity.end.x - entity.start.x), 0))
            ]
        return 0.0, []
    if t == "SYMMETRIC":
        pa = _point_value(sketch, a)
        pb = _point_value(sketch, b)
        axis = sketch.entity(constraint.metadata.get("axis", ""))
        if pa is None or pb is None or not isinstance(axis, LineEntity):
            return 0.0, []
        d = axis.direction()
        if d.norm() < eps:
            return 0.0, []
        d = d / d.norm()
        # reflect a about the axis line through axis.start
        v = pa - axis.start
        proj = d * v.dot(d)
        perp = v - proj
        reflected = axis.start + proj - perp
        delta = reflected - pb
        return delta.norm(), [(a, -delta * 0.5), (b, delta * 0.5)]
    if t == "EQUAL_LENGTH":
        ea = sketch.entity(a)
        eb = sketch.entity(b)
        if isinstance(ea, LineEntity) and isinstance(eb, LineEntity):
            diff = ea.length() - eb.length()
            return abs(diff), [(a, Vec(diff, 0)), (b, Vec(-diff, 0))]
        return 0.0, []
    if t == "EQUAL_RADIUS":
        ea = sketch.entity(a)
        eb = sketch.entity(b)
        if isinstance(ea, (CircleEntity, ArcEntity)) and isinstance(eb, (CircleEntity, ArcEntity)):
            diff = ea.radius - eb.radius
            return abs(diff), [(a, Vec(diff, 0)), (b, Vec(-diff, 0))]
        return 0.0, []
    if t == "PARALLEL":
        ea = sketch.entity(a)
        eb = sketch.entity(b)
        if isinstance(ea, LineEntity) and isinstance(eb, LineEntity):
            cross = abs(ea.direction().cross(eb.direction()).z)
            return cross, [(a, Vec(cross, cross * 0.5)), (b, Vec(-cross, 0))]
        return 0.0, []
    if t == "PERPENDICULAR":
        ea = sketch.entity(a)
        eb = sketch.entity(b)
        if isinstance(ea, LineEntity) and isinstance(eb, LineEntity):
            dot = abs(ea.direction().dot(eb.direction()))
            return dot, [(a, Vec(dot, 0)), (b, Vec(0, dot))]
        return 0.0, []
    return 0.0, []


class SketchConstraintSolver:
    """DOF analysis + numeric constraint solving for a 2D sketch."""

    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 2000) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ DOF
    @staticmethod
    def degrees_of_freedom(sketch: Sketch) -> int:
        dof = 0
        for entity in sketch.entities:
            if entity.is_construction:
                continue
            dof += _ENTITY_DOF.get(entity.kind, 2)
        return dof

    @staticmethod
    def constrained_dof(sketch: Sketch) -> int:
        return sum(c.consumes_dof for c in sketch.constraints) + len(sketch.dimensions)

    @staticmethod
    def analyze_degrees(sketch: Sketch) -> ConstraintSolution:
        """Classify the sketch as under / fully / over constrained."""
        dof = SketchConstraintSolver.degrees_of_freedom(sketch)
        consumed = SketchConstraintSolver.constrained_dof(sketch)
        messages: list[str] = []
        if dof == 0:
            status = "fully"
        elif consumed > dof:
            status = "over"
            messages.append(f"{consumed} DOF consumed vs {dof} available")
        elif consumed >= dof:
            status = "fully"
        else:
            status = "under"
            messages.append(f"{dof - consumed} degrees of freedom remaining")
        return ConstraintSolution(status, max(0, dof - consumed), 0.0, messages=messages)

    # -------------------------------------------------------------- numeric
    def solve(self, sketch: Sketch) -> ConstraintSolution:
        """Solve point-based constraints to reduce the maximum residual."""
        participants = _point_names(sketch)
        for iteration in range(1, self.max_iterations + 1):
            worst = 0.0
            for constraint in sketch.constraints:
                if constraint.constraint_type not in (
                    "COINCIDENT",
                    "CONCENTRIC",
                    "MIDPOINT",
                    "HORIZONTAL",
                    "VERTICAL",
                    "SYMMETRIC",
                    "EQUAL_LENGTH",
                    "PARALLEL",
                    "PERPENDICULAR",
                    "FIXED",
                ):
                    continue
                residual, corrections = _entity_residual(sketch, constraint)
                worst = max(worst, residual)
                if residual <= self.tolerance:
                    continue
                for entity_name, delta in corrections:
                    if entity_name in participants:
                        current = _point_value(sketch, entity_name)
                        if current is not None:
                            _set_point(sketch, entity_name, current + delta)
            if worst <= self.tolerance:
                dof_solution = self.analyze_degrees(sketch)
                return ConstraintSolution(
                    dof_solution.status,
                    dof_solution.dof,
                    worst,
                    iterations=iteration,
                    messages=dof_solution.messages,
                )
        dof_solution = self.analyze_degrees(sketch)
        return ConstraintSolution(
            dof_solution.status,
            dof_solution.dof,
            worst,
            iterations=self.max_iterations,
            messages=[f"max residual {worst:.2e} exceeds tolerance", *dof_solution.messages],
        )

    def is_fully_constrained(self, sketch: Sketch) -> bool:
        return self.analyze_degrees(sketch).is_fully_constrained


def is_parallel_entities(a: LineEntity, b: LineEntity, tol: float = 1e-6) -> bool:
    return abs(a.direction().cross(b.direction()).z) <= tol


def is_perpendicular_entities(a: LineEntity, b: LineEntity, tol: float = 1e-6) -> bool:
    return abs(a.direction().dot(b.direction())) <= tol


def is_tangent(a: SketchEntity, b: SketchEntity, tol: float = 1e-6) -> bool:
    """Geometric tangency test between common sketch entity pairs."""
    if isinstance(a, LineEntity) and isinstance(b, CircleEntity):
        return abs(math.dist((a.start.x, a.start.y), (b.center.x, b.center.y)) - b.radius) <= tol
    if isinstance(b, LineEntity) and isinstance(a, CircleEntity):
        return abs(math.dist((b.start.x, b.start.y), (a.center.x, a.center.y)) - a.radius) <= tol
    if isinstance(a, CircleEntity) and isinstance(b, CircleEntity):
        d = math.dist((a.center.x, a.center.y), (b.center.x, b.center.y))
        return abs(d - (a.radius + b.radius)) <= tol or abs(d - abs(a.radius - b.radius)) <= tol
    return False


__all__ = [
    "CONSTRAINT_TYPES",
    "ConstraintSolution",
    "GeometricConstraint",
    "SketchConstraintSolver",
    "is_parallel_entities",
    "is_perpendicular_entities",
    "is_tangent",
]
