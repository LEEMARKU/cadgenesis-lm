"""cadgenesis.cad.mechanisms.linkages
==================================
Four-bar linkage kinematics.

A four-bar linkage has a ground link, a crank (input), a coupler and a
rocker (output).  This module checks the Grashof condition and computes
coupler and rocker positions for a given crank angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class FourBarLinkage:
    """Planar four-bar linkage: ground, crank, coupler, rocker lengths.

    ``ground_offset`` is the vertical offset between the ground pivots.
    """

    name: str = "fourbar"
    ground: float = 60.0
    crank: float = 20.0
    coupler: float = 70.0
    rocker: float = 40.0
    ground_offset: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.ground, self.crank, self.coupler, self.rocker):
            if value <= 0:
                raise ValueError("link lengths must be positive")

    # -- Grashof condition -------------------------------------------------------
    @property
    def shortest(self) -> float:
        return min(self.ground, self.crank, self.coupler, self.rocker)

    @property
    def longest(self) -> float:
        return max(self.ground, self.crank, self.coupler, self.rocker)

    @property
    def is_grashof(self) -> bool:
        """True when the shortest link can make a full revolution."""
        s = sum((self.ground, self.crank, self.coupler, self.rocker))
        return self.shortest + self.longest <= s - self.shortest - self.longest

    @property
    def mechanism_type(self) -> str:
        """Crank-rocker / double-crank / double-rocker classification."""
        if not self.is_grashof:
            return "double-rocker (non-Grashof)"
        if self.shortest == self.crank:
            return "crank-rocker"
        if self.shortest == self.ground:
            return "double-crank (drag link)"
        return "double-rocker"

    def rocker_angle(self, crank_angle_deg: float) -> float | None:
        """Output rocker angle (degrees) for a given crank angle.

        Returns None when the linkage cannot assemble at that angle.
        """
        theta = math.radians(crank_angle_deg)
        ox = self.ground
        oy = self.ground_offset
        ax = self.crank * math.cos(theta)
        ay = self.crank * math.sin(theta)
        d = math.hypot(ax - ox, ay - oy)
        r1 = self.coupler
        r2 = self.rocker
        if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
            return None
        # angle between line B->O2 and the coupler
        alpha = math.acos((r1**2 + d**2 - r2**2) / (2 * r1 * d))
        base = math.atan2(oy - ay, ox - ax)
        # coupler point B
        bx = ax + r1 * math.cos(base + alpha)
        by = ay + r1 * math.sin(base + alpha)
        rocker_angle = math.degrees(math.atan2(by - oy, bx - ox))
        return rocker_angle % 360.0

    def coupler_point(
        self, crank_angle_deg: float, along_coupler: float = 0.5
    ) -> tuple[float, float] | None:
        """Position of a point along the coupler link (0=at crank, 1=at rocker)."""
        theta = math.radians(crank_angle_deg)
        ax = self.crank * math.cos(theta)
        ay = self.crank * math.sin(theta)
        rocker = self.rocker_angle(crank_angle_deg)
        if rocker is None:
            return None
        beta = math.radians(rocker)
        ox = self.ground
        oy = self.ground_offset
        bx = ox + self.rocker * math.cos(beta)
        by = oy + self.rocker * math.sin(beta)
        return (ax + (bx - ax) * along_coupler, ay + (by - ay) * along_coupler)

    def sweep_angle(self) -> float:
        """Total rocker sweep (degrees) through a full crank rotation."""
        angles = [self.rocker_angle(a) for a in range(0, 361, 5)]
        valid = [a for a in angles if a is not None]
        if not valid:
            return 0.0
        return max(valid) - min(valid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ground": self.ground,
            "crank": self.crank,
            "coupler": self.coupler,
            "rocker": self.rocker,
            "grashof": self.is_grashof,
            "mechanism_type": self.mechanism_type,
        }


__all__ = ["FourBarLinkage"]
