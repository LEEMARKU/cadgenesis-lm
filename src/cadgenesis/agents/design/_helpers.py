"""cadgenesis.agents.design._helpers
===================================
Shared payload builders for the design-swarm agents.

Converts the plain-dictionary task payloads used by :class:`AgentRequest`
into :class:`~cadgenesis.world_model.objects.WorldObject` and
:class:`~cadgenesis.world_model.objects.LoadCase` instances so the FEA agent
can reason with the existing world-model machinery.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.world_model.objects import (
    BoundaryCondition,
    LoadCase,
    Material,
    WorldObject,
    make_object,
)


def build_world_object(raw: Any) -> WorldObject:
    """Build a :class:`WorldObject` from a dict or pass one through."""
    if isinstance(raw, WorldObject):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("'object' must be a WorldObject or a dict")
    material_raw = raw.get("material")
    material: str | Material | None = None
    if isinstance(material_raw, dict):
        material = Material.from_dict(material_raw)
    elif material_raw is not None:
        material = str(material_raw)
    return make_object(
        feature=str(raw.get("feature", "block")),
        name=str(raw.get("name", "part")),
        parameters=dict(raw.get("parameters", {})),
        material=material,
    )


def build_load_cases(raw: Any) -> list[LoadCase]:
    """Convert a list of load-case dicts into :class:`LoadCase` objects."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("'load_cases' must be a list")
    cases: list[LoadCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError("every load case must be a dict")
        conditions = [
            BoundaryCondition(
                kind=str(condition["kind"]),
                magnitude=float(condition.get("magnitude", 0.0)),
                name=str(condition.get("name", "")),
            )
            for condition in item.get("conditions", [])
            if isinstance(condition, dict)
        ]
        cases.append(
            LoadCase(
                name=str(item.get("name", f"load_case_{index + 1}")),
                conditions=conditions,
                safety_factor_target=float(item.get("safety_factor_target", 0.0)),
            )
        )
    return cases


__all__ = ["build_load_cases", "build_world_object"]
