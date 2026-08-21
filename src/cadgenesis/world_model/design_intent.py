"""cadgenesis.world_model.design_intent
=====================================
Design intent capture (Pillar 4).

:class:`DesignIntentCapture` records the *why* behind a design — goals,
requirements (reusing :class:`cadgenesis.reasoning.Constraint`), rationale
and parameter assignments — and can materialize it as an explicit constraint
set that the world model can validate against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.reasoning.constraint_solver import Constraint, ConstraintSolver
from cadgenesis.world_model.objects import WorldObject


@dataclass
class IntentAnnotation:
    """A single design-intent annotation."""

    target: str
    kind: str  # goal | requirement | rationale | note
    text: str
    constraint: Constraint | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "text": self.text,
            "constraint": (
                {
                    "name": self.constraint.name,
                    "terms": dict(self.constraint.terms),
                    "operator": self.constraint.operator,
                    "rhs": self.constraint.rhs,
                    "tolerance": self.constraint.tolerance,
                }
                if self.constraint is not None
                else None
            ),
        }


@dataclass
class DesignIntent:
    """Captured intent for an assembly of world objects."""

    name: str
    goals: list[str] = field(default_factory=list)
    annotations: list[IntentAnnotation] = field(default_factory=list)
    requirements: list[Constraint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "goals": list(self.goals),
            "annotations": [a.to_dict() for a in self.annotations],
            "requirements": [
                {
                    "name": c.name,
                    "terms": dict(c.terms),
                    "operator": c.operator,
                    "rhs": c.rhs,
                    "tolerance": c.tolerance,
                }
                for c in self.requirements
            ],
        }


class DesignIntentCapture:
    """Build and validate design intent."""

    def __init__(self) -> None:
        self._solver = ConstraintSolver()

    # ------------------------------------------------------------- capture

    def capture(
        self,
        name: str,
        goals: list[str] | None = None,
        annotations: list[IntentAnnotation] | None = None,
    ) -> DesignIntent:
        intent = DesignIntent(name=name, goals=list(goals or []))
        for annotation in annotations or []:
            intent.annotations.append(annotation)
            if annotation.constraint is not None:
                intent.requirements.append(annotation.constraint)
        return intent

    @staticmethod
    def envelope_constraint(
        obj: WorldObject,
        width: float,
        height: float,
        depth: float,
        name: str = "envelope",
    ) -> Constraint:
        """Requirement that an object's extents respect an envelope.

        Encode as ``extent / envelope_size <= 1`` for the largest ratio.
        """
        lo, hi = obj.bounds()
        extents = {
            "x": hi.x - lo.x,
            "y": hi.y - lo.y,
            "z": hi.z - lo.z,
        }
        limits = {"x": width, "y": height, "z": depth}
        worst_ratio = max(
            extents[axis] / limits[axis] if limits[axis] > 0 else 1e9 for axis in ("x", "y", "z")
        )
        return Constraint(
            name=name,
            terms={f"{obj.name}_envelope_ratio": 1.0},
            operator="<=",
            rhs=max(1.0, worst_ratio),
        )

    @staticmethod
    def mass_constraint(
        objects: list[WorldObject],
        limit_kg: float,
        name: str = "mass",
    ) -> Constraint:
        """Requirement that a set of objects stays within a mass budget."""
        variables = {f"mass_{o.object_id}": 1.0 for o in objects}
        return Constraint(name=name, terms=variables, operator="<=", rhs=limit_kg)

    # ------------------------------------------------------------- validate

    def validate(
        self,
        intent: DesignIntent,
        assignment: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Evaluate each requirement against an assignment of variable values."""
        results: list[dict[str, Any]] = []
        for constraint in intent.requirements:
            residual = constraint.residual(assignment)
            results.append(
                {
                    "name": constraint.name,
                    "passed": constraint.operator == "==" and abs(residual) <= constraint.tolerance
                    if constraint.operator == "=="
                    else residual <= constraint.tolerance,
                    "residual": residual,
                    "constraint": constraint.name,
                }
            )
        return results

    def assign_parameters(self, obj: WorldObject) -> dict[str, float]:
        """Variable assignment for an object from its parameters."""
        assignment: dict[str, float] = {}
        for key, value in obj.parameters.items():
            if isinstance(value, (int, float)):
                assignment[f"{obj.name}_{key}"] = float(value)
        assignment[f"mass_{obj.object_id}"] = obj.mass()
        return assignment


__all__ = [
    "DesignIntent",
    "DesignIntentCapture",
    "IntentAnnotation",
]
