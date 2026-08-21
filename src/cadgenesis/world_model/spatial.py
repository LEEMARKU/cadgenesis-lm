"""cadgenesis.world_model.spatial
================================
Spatial reasoning (Pillar 4).

The :class:`SpatialReasoner` answers geometric questions about
:class:`~cadgenesis.world_model.objects.WorldObject` poses and extents:
bounds in the world frame, clearance between objects, overlap /
interference, containment (fit), separation distances and symmetry.

All math reuses ``cadgenesis.cad.geometry.core`` (Vec / Transform); the
reasoner only composes it with the object bounds machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Transform, Vec
from cadgenesis.world_model.objects import WorldObject


@dataclass
class SpatialReport:
    """Aggregated spatial-reasoning result."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def add(self, name: str, ok: bool, details: str = "", value: float | None = None) -> None:
        self.checks.append(
            {
                "name": name,
                "passed": ok,
                "details": details,
                "value": value,
            }
        )
        self.passed = self.passed and ok

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "passed_checks": self.passed_checks,
            "total_checks": len(self.checks),
        }


class SpatialReasoner:
    """Geometric reasoning over world-model objects.

    ``tolerance`` is the comparison epsilon (mm) used for all checks.
    """

    def __init__(self, tolerance: float = 1e-4) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance

    # -------------------------------------------------------------- geometry

    def world_bounds(self, obj: WorldObject) -> tuple[Vec, Vec]:
        """AABB of an object in the world frame.

        :meth:`WorldObject.bounds` already centres the local box at the object
        position, so we subtract that centre to recover local half-extents,
        transform the 8 corners by the pose (handles rotation) and take the
        axis-aligned envelope.
        """
        local_min, local_max = obj.bounds()
        assert obj.pose is not None
        center = obj.position
        corners = [
            obj.pose.apply(Vec(px - center.x, py - center.y, pz - center.z))
            for px in (local_min.x, local_max.x)
            for py in (local_min.y, local_max.y)
            for pz in (local_min.z, local_max.z)
        ]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        return Vec(min(xs), min(ys), min(zs)), Vec(max(xs), max(ys), max(zs))

    def overlap(
        self,
        a: WorldObject,
        b: WorldObject,
    ) -> bool:
        """True when the world AABBs of two objects intersect."""
        a_min, a_max = self.world_bounds(a)
        b_min, b_max = self.world_bounds(b)
        return not (
            a_max.x <= b_min.x + self.tolerance
            or b_max.x <= a_min.x + self.tolerance
            or a_max.y <= b_min.y + self.tolerance
            or b_max.y <= a_min.y + self.tolerance
            or a_max.z <= b_min.z + self.tolerance
            or b_max.z <= a_min.z + self.tolerance
        )

    def clearance(
        self,
        a: WorldObject,
        b: WorldObject,
        axis: str = "z",
    ) -> float:
        """Signed separation along ``axis`` (positive = no contact)."""
        a_min, a_max = self.world_bounds(a)
        b_min, b_max = self.world_bounds(b)
        lo_a = getattr(a_min, axis)
        hi_a = getattr(a_max, axis)
        lo_b = getattr(b_min, axis)
        hi_b = getattr(b_max, axis)
        if hi_a <= lo_b:
            return lo_b - hi_a
        if hi_b <= lo_a:
            return lo_a - hi_b
        return -min(hi_a, hi_b) + max(lo_a, lo_b)

    def interference(
        self,
        a: WorldObject,
        b: WorldObject,
        epsilon: float = 1e-3,
    ) -> bool:
        """True when ``a`` and ``b`` penetrate each other (clearance < ``-epsilon``)."""
        return self.clearance(a, b) < -epsilon

    def tangent(
        self,
        a: WorldObject,
        b: WorldObject,
        epsilon: float = 1e-3,
    ) -> bool:
        """True when ``a`` and ``b`` are within ``epsilon`` of contact."""
        g = self.clearance(a, b)
        return -epsilon <= g <= epsilon

    def clearance_report(
        self,
        a: WorldObject,
        b: WorldObject,
        minimum: float,
        axis: str = "z",
    ) -> SpatialReport:
        """Check that ``a`` and ``b`` keep ``minimum`` clearance on ``axis``."""
        report = SpatialReport()
        gap = self.clearance(a, b, axis)
        ok = gap >= minimum - self.tolerance
        report.add(
            f"clearance.{axis}",
            ok,
            details=f"gap={gap:.4f} mm required={minimum}",
            value=gap,
        )
        if self.overlap(a, b):
            report.add("overlap", False, details="AABBs intersect")
        return report

    def fits_inside(
        self,
        outer: WorldObject,
        inner: WorldObject,
    ) -> bool:
        """True when ``inner``'s world AABB is fully inside ``outer``'s."""
        o_min, o_max = self.world_bounds(outer)
        i_min, i_max = self.world_bounds(inner)
        return (
            i_min.x >= o_min.x - self.tolerance
            and i_max.x <= o_max.x + self.tolerance
            and i_min.y >= o_min.y - self.tolerance
            and i_max.y <= o_max.y + self.tolerance
            and i_min.z >= o_min.z - self.tolerance
            and i_max.z <= o_max.z + self.tolerance
        )

    def distance_between(
        self,
        a: WorldObject,
        b: WorldObject,
    ) -> float:
        """Euclidean distance between object centers."""
        return a.position.distance_to(b.position)

    def relative_pose(
        self,
        a: WorldObject,
        b: WorldObject,
    ) -> Transform:
        """Transform that maps ``b``'s frame into ``a``'s frame."""
        assert a.pose is not None and b.pose is not None
        return a.pose.inverted().composed(b.pose)

    def is_symmetric(
        self,
        obj: WorldObject,
        plane: str = "xz",
    ) -> bool:
        """Best-effort symmetry check on a centered object.

        A primitive is "symmetric" about plane ``plane`` when its extents in
        the two in-plane axes are equal (e.g. a block of 100x60 is NOT xz
        symmetric, 100x100 is).  Centering is inferred from the pose.
        """
        local_min, local_max = obj.bounds()
        widths = {
            "xy": (local_max.x - local_min.x, local_max.y - local_min.y),
            "xz": (local_max.x - local_min.x, local_max.z - local_min.z),
            "yz": (local_max.y - local_min.y, local_max.z - local_min.z),
        }
        if plane not in widths:
            raise ValueError(f"plane must be one of {list(widths)}")
        u, v = widths[plane]
        return abs(u - v) <= self.tolerance


__all__ = ["SpatialReasoner", "SpatialReport"]
