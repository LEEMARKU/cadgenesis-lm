"""cadgenesis.cad.mechanisms.cams
===============================
Cam profile generation from a follower displacement diagram.

Supports the standard follower motions (rise / dwell / fall) with simple
harmonic or cycloidal displacement laws, and produces the cam contour as a
polar list of (angle, radius) points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MOTION_LAWS = ("harmonic", "cycloidal", "constant")


@dataclass
class CamSegment:
    """One motion segment of the displacement diagram."""

    start_angle: float  # degrees
    end_angle: float
    rise: float  # follower lift (mm); negative = fall
    law: str = "harmonic"

    def __post_init__(self) -> None:
        if self.law not in MOTION_LAWS:
            raise ValueError(f"unknown motion law {self.law!r}; expected one of {MOTION_LAWS}")
        if self.end_angle <= self.start_angle:
            raise ValueError("segment end angle must exceed start angle")

    @property
    def span_degrees(self) -> float:
        return self.end_angle - self.start_angle

    def displacement(self, angle: float) -> float:
        """Follower displacement (mm) at a cam angle (degrees) within this segment."""
        t = (angle - self.start_angle) / self.span_degrees
        t = max(0.0, min(1.0, t))
        lift = self.rise
        if self.law == "constant":
            return lift * t
        if self.law == "cycloidal":
            return lift * (t - math.sin(2 * math.pi * t) / (2 * math.pi))
        # simple harmonic motion (SHM)
        return lift * (1.0 - math.cos(math.pi * t)) / 2.0


@dataclass
class CamProfile:
    """A cam defined by base radius and a displacement diagram."""

    base_radius: float
    segments: list[CamSegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.base_radius <= 0:
            raise ValueError("base radius must be positive")

    def add_segment(self, segment: CamSegment) -> CamSegment:
        if self.segments and segment.start_angle < self.segments[-1].end_angle:
            raise ValueError("cam segments must not overlap")
        self.segments.append(segment)
        return segment

    def add_rise_dwell_fall(
        self,
        rise: float,
        rise_span: float,
        dwell_span: float,
        fall_span: float,
        law: str = "harmonic",
        start_angle: float = 0.0,
    ) -> None:
        """Convenience: rise -> dwell -> fall covering 360 degrees."""
        rise_start = start_angle
        dwell_start = rise_start + rise_span
        fall_start = dwell_start + dwell_span
        fall_end = fall_start + fall_span
        self.add_segment(CamSegment(rise_start, dwell_start, rise, law))
        self.add_segment(CamSegment(dwell_start, fall_start, 0.0, "constant"))
        self.add_segment(CamSegment(fall_start, fall_end, -rise, law))

    def displacement_at(self, angle_deg: float) -> float:
        angle = angle_deg % 360.0
        for segment in self.segments:
            if segment.start_angle <= angle <= segment.end_angle:
                return segment.displacement(angle)
        return 0.0

    def max_rise(self) -> float:
        return max((s.rise for s in self.segments), default=0.0)

    def pitch_radius_at(self, angle_deg: float) -> float:
        """Distance from cam centre to the follower centre (pitch curve)."""
        return self.base_radius + self.displacement_at(angle_deg)

    def cam_radius_at(self, angle_deg: float) -> float:
        """Distance from cam centre to the cam surface (roller follower)."""
        return self.pitch_radius_at(angle_deg)

    def profile_points(self, samples: int = 72) -> list[tuple[float, float]]:
        """Cam contour in cartesian coordinates (x, y) for ``samples`` angles."""
        points: list[tuple[float, float]] = []
        for i in range(samples):
            angle = math.radians(360.0 * i / samples)
            radius = self.cam_radius_at(360.0 * i / samples)
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
        return points

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_radius": self.base_radius,
            "segments": [
                {
                    "start_angle": s.start_angle,
                    "end_angle": s.end_angle,
                    "rise": s.rise,
                    "law": s.law,
                }
                for s in self.segments
            ],
            "max_rise": self.max_rise(),
        }


__all__ = ["MOTION_LAWS", "CamProfile", "CamSegment"]
