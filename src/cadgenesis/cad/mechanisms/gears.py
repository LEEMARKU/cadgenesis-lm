"""cadgenesis.cad.mechanisms.gears
===============================
Spur gear geometry and gear-train analysis.

Implements standard involute spur gear parameters (module, teeth, pitch
diameter, addendum/dedendum, base circle), involute tooth-profile sampling,
gear ratios and compound gear trains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PRESSURE_ANGLE_DEFAULT = 20.0  # degrees


@dataclass
class SpurGear:
    """A standard involute spur gear."""

    name: str
    module: float  # mm
    teeth: int
    pressure_angle_deg: float = PRESSURE_ANGLE_DEFAULT
    face_width: float = 10.0

    def __post_init__(self) -> None:
        if self.module <= 0:
            raise ValueError("module must be positive")
        if self.teeth < 5:
            raise ValueError("a gear needs at least 5 teeth")

    # -- derived geometry -------------------------------------------------------
    @property
    def pitch_diameter(self) -> float:
        return self.module * self.teeth

    @property
    def pitch_radius(self) -> float:
        return self.pitch_diameter / 2.0

    @property
    def base_diameter(self) -> float:
        return self.pitch_diameter * math.cos(math.radians(self.pressure_angle_deg))

    @property
    def base_radius(self) -> float:
        return self.base_diameter / 2.0

    @property
    def addendum(self) -> float:
        return self.module

    @property
    def dedendum(self) -> float:
        return 1.25 * self.module

    @property
    def outer_diameter(self) -> float:
        return self.pitch_diameter + 2 * self.addendum

    @property
    def root_diameter(self) -> float:
        return self.pitch_diameter - 2 * self.dedendum

    @property
    def circular_pitch(self) -> float:
        return math.pi * self.module

    @property
    def tooth_thickness(self) -> float:
        return math.pi * self.module / 2.0

    # -- involute profile ---------------------------------------------------------
    def involute_points(self, samples: int = 12) -> list[tuple[float, float]]:
        """Points on one involute tooth flank (starting at the base circle).

        Returns a list of (x, y) in the gear plane; mirror about the tooth
        centre line to build a full tooth profile.
        """
        rb = self.base_radius
        ro = self.outer_diameter / 2.0
        points: list[tuple[float, float]] = []
        # involute from base radius to outer radius
        for i in range(samples + 1):
            t = (i / samples) * math.sqrt(max(0.0, (ro / rb) ** 2 - 1.0))
            x = rb * (math.cos(t) + t * math.sin(t))
            y = rb * (math.sin(t) - t * math.cos(t))
            points.append((x, y))
        return points

    def tooth_points(self, samples: int = 12) -> list[tuple[float, float]]:
        """A complete single-tooth outline (involute flank, tip arc, back flank)."""
        flank = self.involute_points(samples)
        # mirror flank across the x-axis to form the other side
        mirrored = [(-x, y) for x, y in reversed(flank)]
        return flank + mirrored

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "teeth": self.teeth,
            "pressure_angle_deg": self.pressure_angle_deg,
            "face_width": self.face_width,
            "pitch_diameter": self.pitch_diameter,
            "outer_diameter": self.outer_diameter,
        }


@dataclass
class GearPair:
    """Two meshing gears with a centre distance and ratio."""

    driver: SpurGear
    driven: SpurGear

    @property
    def ratio(self) -> float:
        """Velocity ratio driver:driven (n_driver / n_driven)."""
        return self.driven.teeth / self.driver.teeth

    @property
    def centre_distance(self) -> float:
        return (self.driver.pitch_diameter + self.driven.pitch_diameter) / 2.0

    @property
    def output_speed_factor(self) -> float:
        return self.driver.teeth / self.driven.teeth

    def check_mesh(self, backlash_mm: float = 0.0) -> bool:
        """True when centre distance matches the pitch-line requirement."""
        return (
            abs(
                self.centre_distance
                - (self.driver.pitch_radius + self.driven.pitch_radius + backlash_mm)
            )
            <= 1e-9
        )


def gear_ratio(driver_teeth: int, driven_teeth: int) -> float:
    """Convenience ratio helper (driven / driver)."""
    if driver_teeth <= 0 or driven_teeth <= 0:
        raise ValueError("tooth counts must be positive")
    return driven_teeth / driver_teeth


class GearTrain:
    """A compound gear train; total ratio is the product of stage ratios."""

    def __init__(self) -> None:
        self.stages: list[GearPair] = []

    def add_stage(self, driver: SpurGear, driven: SpurGear) -> GearPair:
        pair = GearPair(driver, driven)
        self.stages.append(pair)
        return pair

    def total_ratio(self) -> float:
        """Compound ratio (output speed relative to input)."""
        product = 1.0
        for pair in self.stages:
            product *= pair.output_speed_factor
        return product

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": len(self.stages),
            "total_ratio": self.total_ratio(),
            "pairs": [{"driver": p.driver.name, "driven": p.driven.name} for p in self.stages],
        }


__all__ = ["GearPair", "GearTrain", "SpurGear", "gear_ratio"]
