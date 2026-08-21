"""cadgenesis.cad.mechanisms.parts
===============================
Standard machine parts: bearings and shafts.

These are lightweight parametric descriptors (dimensions + fit/tolerance
data) used by assembly and manufacturing reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BEARING_TYPES = ("ball_radial", "ball_angular", "roller_radial", "thrust", "needle", "bushing")


@dataclass
class Bearing:
    """A standard rolling-element bearing descriptor."""

    name: str
    bearing_type: str
    bore_mm: float
    outer_diameter_mm: float
    width_mm: float
    load_rating_kn: float = 0.0

    def __post_init__(self) -> None:
        if self.bearing_type not in BEARING_TYPES:
            raise ValueError(f"unknown bearing type {self.bearing_type!r}")
        if self.bore_mm <= 0 or self.outer_diameter_mm <= self.bore_mm or self.width_mm <= 0:
            raise ValueError("invalid bearing dimensions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bearing_type": self.bearing_type,
            "bore_mm": self.bore_mm,
            "outer_diameter_mm": self.outer_diameter_mm,
            "width_mm": self.width_mm,
            "load_rating_kn": self.load_rating_kn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bearing:
        return cls(
            name=str(data["name"]),
            bearing_type=str(data["bearing_type"]),
            bore_mm=float(data["bore_mm"]),
            outer_diameter_mm=float(data["outer_diameter_mm"]),
            width_mm=float(data["width_mm"]),
            load_rating_kn=float(data.get("load_rating_kn", 0.0)),
        )


@dataclass
class Shaft:
    """A rotating shaft descriptor with journal locations."""

    name: str
    diameter_mm: float
    length_mm: float
    material: str = "AISI 1045"
    journal_positions_mm: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.diameter_mm <= 0 or self.length_mm <= 0:
            raise ValueError("shaft dimensions must be positive")
        for position in self.journal_positions_mm:
            if not 0 <= position <= self.length_mm:
                raise ValueError("journal position must lie within the shaft length")

    def volume_mm3(self) -> float:
        import math

        return math.pi * (self.diameter_mm / 2) ** 2 * self.length_mm

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "diameter_mm": self.diameter_mm,
            "length_mm": self.length_mm,
            "material": self.material,
            "journal_positions_mm": list(self.journal_positions_mm),
        }


__all__ = ["BEARING_TYPES", "Bearing", "Shaft"]
