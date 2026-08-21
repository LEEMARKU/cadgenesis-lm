"""cadgenesis.cad.manufacturing.features
=====================================
Manufacturing feature recognition and description.

Manufacturing features describe *how a part is physically produced*: CNC
machining, 3D printing, casting, injection moulding, sheet metal and welding.
Each feature is a named operation with parameters, a link to the CAD feature
it maps from, and validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROCESS_GROUPS = (
    "cnc",
    "3d_printing",
    "casting",
    "injection_molding",
    "sheet_metal",
    "welding",
)


@dataclass
class ManufacturingFeature:
    """A manufacturing operation on a part."""

    name: str
    process_group: str
    operation: str
    params: dict[str, Any] = field(default_factory=dict)
    source_feature: str = ""  # reference to the CAD feature it implements
    materials: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.process_group not in PROCESS_GROUPS:
            raise ValueError(
                f"unknown process group {self.process_group!r}; expected one of {PROCESS_GROUPS}"
            )
        if not self.name or not self.operation:
            raise ValueError("manufacturing feature needs a name and operation")

    def validate(self) -> list[str]:
        """Generic parameter sanity checks; subclasses extend."""
        problems: list[str] = []
        for key, value in self.params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value != value:
                problems.append(f"parameter {key!r} is NaN")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "process_group": self.process_group,
            "operation": self.operation,
            "params": dict(self.params),
            "source_feature": self.source_feature,
            "materials": list(self.materials),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManufacturingFeature:
        return cls(
            name=str(data["name"]),
            process_group=str(data["process_group"]),
            operation=str(data["operation"]),
            params=data.get("params") or {},
            source_feature=str(data.get("source_feature", "")),
            materials=[str(m) for m in data.get("materials", [])],
        )


def make_feature(
    process_group: str, operation: str, name: str = "", **params: Any
) -> ManufacturingFeature:
    """Factory helper for manufacturing features."""
    return ManufacturingFeature(
        name=name or f"{process_group}_{operation.lower().replace(' ', '_')}",
        process_group=process_group,
        operation=operation,
        params=params,
    )


# ---------------------------------------------------------------------------
# Convenience builders per process group
# ---------------------------------------------------------------------------


def cnc_feature(operation: str, **params: Any) -> ManufacturingFeature:
    return make_feature("cnc", operation, **params)


def print_feature(operation: str, **params: Any) -> ManufacturingFeature:
    return make_feature("3d_printing", operation, **params)


def casting_feature(operation: str, **params: Any) -> ManufacturingFeature:
    return make_feature("casting", operation, **params)


def injection_feature(operation: str, **params: Any) -> ManufacturingFeature:
    return make_feature("injection_molding", operation, **params)


def sheet_metal_feature(operation: str, **params: Any) -> ManufacturingFeature:
    return make_feature("sheet_metal", operation, **params)


def welding_feature(operation: str, **params: Any) -> ManufacturingFeature:
    return make_feature("welding", operation, **params)


__all__ = [
    "PROCESS_GROUPS",
    "ManufacturingFeature",
    "casting_feature",
    "cnc_feature",
    "injection_feature",
    "make_feature",
    "print_feature",
    "sheet_metal_feature",
    "welding_feature",
]
