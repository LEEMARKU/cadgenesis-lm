"""cadgenesis.cad.assembly.mates
=============================
Assembly mates, constraints and references with degree-of-freedom (DOF)
analysis.

Mates model how parts relate in an assembly: coincident / flush / parallel /
perpendicular / tangent / concentric / distance / angle / gear / rack-pinion
etc.  Each mate removes a number of degrees of freedom, and the remaining
DOF of a component (or of the whole assembly) is computed from its mates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MATE_TYPES = (
    "COINCIDENT",
    "FLUSH",
    "PARALLEL",
    "PERPENDICULAR",
    "TANGENT",
    "CONCENTRIC",
    "DISTANCE",
    "ANGLE",
    "GEAR",
    "RACK_PINION",
    "SCREW",
    "CAM",
    "SLOT_JOINT",
    "BALL_JOINT",
    "PIVOT",
    "SLIDER",
    "RIGID",
    "SPRING",
    "BELT",
    "WIRE",
)

# DOF removed per mate type (out of the 6 rigid-body DOF for a pair)
_MATE_DOF: dict[str, int] = {
    "COINCIDENT": 3,
    "FLUSH": 3,
    "PARALLEL": 2,
    "PERPENDICULAR": 2,
    "TANGENT": 2,
    "CONCENTRIC": 4,
    "DISTANCE": 3,
    "ANGLE": 2,
    "GEAR": 5,
    "RACK_PINION": 5,
    "SCREW": 5,
    "CAM": 5,
    "SLOT_JOINT": 3,
    "BALL_JOINT": 3,
    "PIVOT": 5,
    "SLIDER": 5,
    "RIGID": 6,
    "SPRING": 4,
    "BELT": 5,
    "WIRE": 4,
}


@dataclass
class Reference:
    """A named reference to a component entity (face / edge / axis / point)."""

    component: str
    entity: str  # face/edge/axis/point identifier within the component
    geometry: str = "face"  # "face" | "edge" | "axis" | "point" | "datum"

    def to_dict(self) -> dict[str, Any]:
        return {"component": self.component, "entity": self.entity, "geometry": self.geometry}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reference:
        return cls(
            component=str(data["component"]),
            entity=str(data["entity"]),
            geometry=str(data.get("geometry", "face")),
        )


@dataclass
class AssemblyConstraint:
    """A named mate between two component references."""

    name: str
    mate_type: str
    reference_a: Reference
    reference_b: Reference
    offset: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mate_type not in MATE_TYPES:
            raise ValueError(f"unknown mate type {self.mate_type!r}; expected one of {MATE_TYPES}")
        if not self.name:
            raise ValueError("assembly constraint name must be non-empty")

    @property
    def removes_dof(self) -> int:
        return _MATE_DOF[self.mate_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mate_type": self.mate_type,
            "reference_a": self.reference_a.to_dict(),
            "reference_b": self.reference_b.to_dict(),
            "offset": self.offset,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyConstraint:
        return cls(
            name=str(data["name"]),
            mate_type=str(data["mate_type"]),
            reference_a=Reference.from_dict(data["reference_a"]),
            reference_b=Reference.from_dict(data["reference_b"]),
            offset=float(data.get("offset", 0.0)),
            parameters=data.get("parameters") or {},
        )


@dataclass
class MateAnalysis:
    """Degree-of-freedom analysis of a component within an assembly."""

    component: str
    dof: int
    dof_removed: int
    mates: list[str] = field(default_factory=list)

    @property
    def is_fully_constrained(self) -> bool:
        return self.dof <= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "dof": self.dof,
            "dof_removed": self.dof_removed,
            "mates": self.mates,
        }


class MateSolver:
    """Computes residual degrees of freedom from assembly constraints."""

    MAX_DOF = 6

    def analyze_component(
        self,
        component_name: str,
        constraints: list[AssemblyConstraint],
    ) -> MateAnalysis:
        """Residual DOF for ``component_name`` given its mates."""
        participating = [
            c
            for c in constraints
            if c.reference_a.component == component_name
            or c.reference_b.component == component_name
        ]
        removed = min(self.MAX_DOF, sum(c.removes_dof for c in participating))
        return MateAnalysis(
            component=component_name,
            dof=max(0, self.MAX_DOF - removed),
            dof_removed=removed,
            mates=[c.name for c in participating],
        )

    def analyze_assembly(
        self,
        component_names: list[str],
        constraints: list[AssemblyConstraint],
    ) -> list[MateAnalysis]:
        return [self.analyze_component(name, constraints) for name in component_names]

    def total_dof(
        self,
        component_names: list[str],
        constraints: list[AssemblyConstraint],
    ) -> int:
        analyses = self.analyze_assembly(component_names, constraints)
        return sum(a.dof for a in analyses)

    def is_rigid(self, component_names: list[str], constraints: list[AssemblyConstraint]) -> bool:
        """True when the ground-referenced assembly has zero residual DOF."""
        # count unique components participating in mates
        named = set(component_names)
        grounded = [
            c
            for c in constraints
            if c.reference_a.component in named and c.reference_b.component in named
        ]
        if not grounded:
            return False
        remaining = self.total_dof(list(named), constraints)
        return remaining <= 0


__all__ = [
    "MATE_TYPES",
    "AssemblyConstraint",
    "MateAnalysis",
    "MateSolver",
    "Reference",
]
