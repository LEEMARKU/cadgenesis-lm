"""cadgenesis.cad.integration.reasoning_bridge
============================================
Convert CAD package objects into the dict structures consumed by the
existing reasoning toolkit (``GeometryReasoner``, ``ConstraintSolver``,
``ManufacturingRules``, ``TopologyAnalyzer``) and back.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.reasoning.constraint_solver import Constraint, Variable
from cadgenesis.reasoning.geometry_reasoner import Primitive


class ReasoningBridge:
    """Two-way translation between CAD objects and reasoning-toolkit inputs."""

    # -- CAD -> reasoning --------------------------------------------------------

    @staticmethod
    def to_primitive(solid: Any, name: str = "") -> Primitive:
        """Convert a ``SolidPrimitive`` into a reasoning ``Primitive``."""
        if isinstance(solid, dict):
            return Primitive(
                kind=str(solid.get("kind", "box")),
                dims={str(k): float(v) for k, v in solid.get("dims", {}).items()},
                name=name or solid.get("name", ""),
            )
        if not hasattr(solid, "to_dict"):
            raise TypeError(f"unsupported solid type {type(solid).__name__}")
        data = solid.to_dict()
        position = data.get("position") or [0.0, 0.0, 0.0]
        return Primitive(
            kind=str(data["kind"]),
            dims={str(k): float(v) for k, v in data.get("dims", {}).items()},
            position=tuple(position) if len(position) == 3 else None,
            name=name or getattr(solid, "name", ""),
        )

    @staticmethod
    def to_variables(
        values: dict[str, float],
        bounds: dict[str, tuple[float, float] | None] | None = None,
    ) -> list[Variable]:
        """Convert a ``{name: value}`` dict into bounded ``Variable`` objects."""
        bounds = bounds or {}
        variables: list[Variable] = []
        for name, value in values.items():
            lo, hi = bounds.get(name, (None, None)) or (None, None)
            variables.append(Variable(name, initial=float(value), lower=lo, upper=hi))
        return variables

    @staticmethod
    def to_constraints(rules: list[dict[str, Any]]) -> list[Constraint]:
        """Convert rule dicts ``{"name", "terms", "operator", "rhs"}`` to Constraints."""
        return [
            Constraint(
                name=str(rule["name"]),
                terms={str(k): float(v) for k, v in rule["terms"].items()},
                operator=str(rule["operator"]),
                rhs=float(rule["rhs"]),
                tolerance=float(rule.get("tolerance", 1e-6)),
            )
            for rule in rules
        ]

    @staticmethod
    def to_manufacturing_part(
        material: str | None = None,
        wall_thickness: float | None = None,
        hole_diameter: float | None = None,
        hole_depth: float | None = None,
        processes: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a DFM ``part`` dict for ``ManufacturingRules.assess``."""
        part: dict[str, Any] = {}
        if material:
            part["material"] = material
        if wall_thickness is not None:
            part["wall_thickness"] = wall_thickness
            part["min_wall_thickness"] = wall_thickness
        if hole_diameter is not None:
            part["hole_diameter"] = hole_diameter
        if hole_depth is not None:
            part["hole_depth"] = hole_depth
        if processes:
            part["processes"] = list(processes)
        if extra:
            part.update(extra)
        return part

    @staticmethod
    def to_topology_stats(solid: Any) -> dict[str, Any]:
        """Extract a topology counts dict from a B-Rep solid for the analyzer."""
        if hasattr(solid, "analyze"):
            data = solid.analyze()
        elif hasattr(solid, "to_dict"):
            data = solid.to_dict().get("analysis", solid.to_dict())
        else:
            data = solid
        return {
            "vertices": int(data.get("vertices", 0)),
            "edges": int(data.get("edges", 0)),
            "faces": int(data.get("faces", 0)),
            "shells": int(data.get("shells", 0)),
            "solids": int(data.get("solids", 1)),
            "loops": int(data.get("loops", 0)),
        }

    # -- reasoning -> CAD ---------------------------------------------------------

    @staticmethod
    def primitives_to_design(primitives: list[Primitive]) -> dict[str, Any]:
        return {
            "primitives": [
                {
                    "kind": p.kind,
                    "dims": dict(p.dims),
                    "position": p.position or (0.0, 0.0, 0.0),
                }
                for p in primitives
            ]
        }


#: Convenience instance for stateless access.
bridge = ReasoningBridge()

__all__ = ["ReasoningBridge", "bridge"]
