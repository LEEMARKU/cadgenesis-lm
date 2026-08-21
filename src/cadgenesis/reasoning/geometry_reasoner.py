"""cadgenesis.reasoning.geometry_reasoner
========================================
Geometric reasoning over CAD primitives: volumes, axis-aligned bounding boxes,
interference / clearance / fit checks and primitive validation.

Primitives are described by a ``kind`` plus a dimension map, e.g.
``{"kind": "box", "dims": {"length": 10.0, "width": 4.0, "height": 3.0}}``.
Positions are optional (x, y, z) triplets used for spatial checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_VALID_KINDS = ("box", "cylinder", "sphere", "cone", "torus", "prism", "pyramid")

# Default dimension names per primitive kind (used to validate + compute).
_KIND_REQUIRED_DIMS: dict[str, tuple[str, ...]] = {
    "box": ("length", "width", "height"),
    "cylinder": ("radius", "height"),
    "sphere": ("radius",),
    "cone": ("radius", "height"),
    "torus": ("major_radius", "minor_radius"),
    "prism": ("base_area", "height"),
    "pyramid": ("base_area", "height"),
}


@dataclass
class Primitive:
    """A geometric primitive with dimensions and optional position."""

    kind: str
    dims: dict[str, float]
    position: tuple[float, float, float] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"unsupported primitive kind {self.kind!r}; expected one of {_VALID_KINDS}"
            )
        for key, value in self.dims.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"dimension {key!r} must be a number")
        if self.position is not None:
            if len(self.position) != 3:
                raise ValueError("position must be an (x, y, z) triple")
            px, py, pz = self.position
            self.position = (float(px), float(py), float(pz))


@dataclass
class GeometryValidation:
    """Outcome of validating a primitive or a group of primitives."""

    valid: bool
    messages: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.valid


class GeometryReasoner:
    """Pure-Python geometry calculations and spatial predicates."""

    # ------------------------------------------------------------- dimensions

    @staticmethod
    def validate(primitive: Primitive) -> GeometryValidation:
        """Check that a primitive has all required, positive dimensions."""
        messages: list[str] = []
        required = _KIND_REQUIRED_DIMS[primitive.kind]
        for dim in required:
            if dim not in primitive.dims:
                messages.append(
                    f"{primitive.name or primitive.kind!r} is missing required dimension {dim!r}"
                )
                continue
            value = primitive.dims[dim]
            if not math.isfinite(float(value)):
                messages.append(f"dimension {dim!r} must be finite")
            elif value <= 0:
                messages.append(f"dimension {dim!r} must be positive, got {value}")
        return GeometryValidation(valid=not messages, messages=messages)

    @staticmethod
    def volume(primitive: Primitive) -> float:
        """Analytical volume of a primitive (raises on invalid input)."""
        check = GeometryReasoner.validate(primitive)
        if not check.valid:
            raise ValueError("; ".join(check.messages))
        dims = primitive.dims
        kind = primitive.kind
        if kind == "box":
            return float(dims["length"] * dims["width"] * dims["height"])
        if kind == "cylinder":
            return math.pi * dims["radius"] ** 2 * dims["height"]
        if kind == "sphere":
            return (4.0 / 3.0) * math.pi * dims["radius"] ** 3
        if kind == "cone":
            return (1.0 / 3.0) * math.pi * dims["radius"] ** 2 * dims["height"]
        if kind == "torus":
            return 2.0 * math.pi**2 * dims["major_radius"] * dims["minor_radius"] ** 2
        if kind == "prism":
            return float(dims["base_area"] * dims["height"])
        return (1.0 / 3.0) * float(dims["base_area"] * dims["height"])

    @staticmethod
    def aabb(primitive: Primitive) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Axis-aligned bounding box as ``((min_x, min_y, min_z), (max_x, ...))``."""
        check = GeometryReasoner.validate(primitive)
        if not check.valid:
            raise ValueError("; ".join(check.messages))
        dims = primitive.dims
        kind = primitive.kind
        if kind == "box":
            extents = (dims["length"], dims["width"], dims["height"])
        elif kind in ("cylinder", "cone"):
            radius = dims["radius"]
            extents = (2 * radius, 2 * radius, dims["height"])
        elif kind == "sphere":
            diameter = 2 * dims["radius"]
            extents = (diameter, diameter, diameter)
        elif kind == "torus":
            diameter = 2 * (dims["major_radius"] + dims["minor_radius"])
            extents = (diameter, diameter, 2 * dims["minor_radius"])
        else:  # prism / pyramid — use base_area side approximation (square)
            side = math.sqrt(dims["base_area"])
            extents = (side, side, dims["height"])

        half = tuple(e / 2.0 for e in extents)
        center = primitive.position or (0.0, 0.0, 0.0)
        return (
            (center[0] - half[0], center[1] - half[1], center[2] - half[2]),
            (center[0] + half[0], center[1] + half[1], center[2] + half[2]),
        )

    # ------------------------------------------------------------- predicates

    @classmethod
    def overlaps(cls, a: Primitive, b: Primitive) -> bool:
        """True if the AABBs of ``a`` and ``b`` overlap (interference test)."""
        (amin, amax) = cls.aabb(a)
        (bmin, bmax) = cls.aabb(b)
        return all(not (amax[i] < bmin[i] or bmax[i] < amin[i]) for i in range(3))

    @classmethod
    def clearance(cls, a: Primitive, b: Primitive) -> float:
        """Signed separation between AABBs along each axis.

        Returns the minimum gap across the three axes; a negative value means
        the boxes overlap by that amount (interference depth).
        """
        (amin, amax) = cls.aabb(a)
        (bmin, bmax) = cls.aabb(b)
        gaps: list[float] = []
        for i in range(3):
            if amax[i] < bmin[i]:
                gaps.append(bmin[i] - amax[i])
            elif bmax[i] < amin[i]:
                gaps.append(amin[i] - bmax[i])
            else:
                gaps.append(min(amax[i] - bmin[i], bmax[i] - amin[i]))
        return float(min(gaps))

    @classmethod
    def check_clearance(cls, a: Primitive, b: Primitive, gap: float) -> bool:
        """True if the AABBs are separated by at least ``gap`` on every axis."""
        return cls.clearance(a, b) >= gap

    @classmethod
    def contains(cls, inner: Primitive, outer: Primitive) -> bool:
        """True if the AABB of ``inner`` fits inside the AABB of ``outer``."""
        (imin, imax) = cls.aabb(inner)
        (omin, omax) = cls.aabb(outer)
        return all(imin[i] >= omin[i] and imax[i] <= omax[i] for i in range(3))

    @classmethod
    def check_fit(cls, part: Primitive, cavity: Primitive) -> bool:
        """Alias for :meth:`contains` (does ``part`` fit inside ``cavity``)."""
        return cls.contains(part, cavity)

    @classmethod
    def combined_bounds(
        cls, primitives: list[Primitive]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Union AABB across many primitives (None for an empty list)."""
        if not primitives:
            return None
        mins: list[float] = [math.inf, math.inf, math.inf]
        maxs: list[float] = [-math.inf, -math.inf, -math.inf]
        for primitive in primitives:
            (pmin, pmax) = cls.aabb(primitive)
            for i in range(3):
                mins[i] = min(mins[i], pmin[i])
                maxs[i] = max(maxs[i], pmax[i])
        return (
            (mins[0], mins[1], mins[2]),
            (maxs[0], maxs[1], maxs[2]),
        )

    # ---------------------------------------------- P7 feature reasoning

    @staticmethod
    def feature_dependencies(
        features: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Ordered (parent, child) feature pairs from ``depends_on`` keys."""
        by_id = {f.get("id"): f for f in features}
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for feature in features:
            for dep in feature.get("depends_on") or []:
                parent = by_id.get(dep)
                if parent is not None:
                    pairs.append((parent, feature))
        return pairs

    @classmethod
    def validate_feature_dependencies(
        cls,
        features: list[dict[str, Any]],
    ) -> GeometryValidation:
        """Check feature dependency edges: dangling refs and cycles.

        A feature dependency graph must reference only known features and be
        acyclic (parents must be created before children).
        """
        messages: list[str] = []
        by_id: dict[str, dict[str, Any]] = {str(f.get("id")): f for f in features}
        ids = set(by_id)
        messages.extend(
            f"feature {feature.get('id')!r} depends on unknown feature {dep!r}"
            for feature in features
            for dep in (feature.get("depends_on") or [])
            if dep not in ids
        )
        visited: dict[str, int] = {}
        order: list[str] = []

        def _visit(node: str) -> bool:
            state = visited.get(node, 0)
            if state == 1:
                return False  # back edge -> cycle
            if state == 2:
                return True
            visited[node] = 1
            feature = by_id[node]
            for dep in feature.get("depends_on") or []:
                if isinstance(dep, str) and dep in by_id and not _visit(dep):
                    return False
            visited[node] = 2
            order.append(node)
            return True

        for node in by_id:
            if not _visit(node):
                messages.append("feature dependency cycle detected")
                break
        return GeometryValidation(valid=not messages, messages=messages)

    @classmethod
    def feature_order(
        cls,
        features: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Topological creation order of features, or ``(False, cycle)``."""
        validation = cls.validate_feature_dependencies(features)
        if not validation.valid:
            return False, []
        by_id: dict[str, dict[str, Any]] = {str(f.get("id")): f for f in features}
        visited: dict[str, int] = {}
        order: list[str] = []

        def _visit(node: str) -> None:
            if visited.get(node, 0) == 2:
                return
            visited[node] = 1
            for dep in by_id[node].get("depends_on") or []:
                if isinstance(dep, str) and dep in by_id:
                    _visit(dep)
            visited[node] = 2
            order.append(node)

        for node in by_id:
            _visit(node)
        return True, order

    @classmethod
    def geometric_consistency(
        cls,
        primitives: list[Primitive],
        allowed_interference: float = 0.0,
    ) -> GeometryValidation:
        """Flag interfering primitive pairs (AABB overlap) in a design.

        ``allowed_interference`` permits overlapping pairs whose interference
        depth stays within the tolerance (e.g. press fits); deeper
        interference is reported as a consistency error.
        """
        messages: list[str] = []
        for i, left in enumerate(primitives):
            for right in primitives[i + 1 :]:
                if cls.overlaps(left, right):
                    depth = cls.clearance(left, right)
                    if depth > allowed_interference:
                        messages.append(
                            f"{left.name or left.kind} overlaps "
                            f"{right.name or right.kind} "
                            f"by {depth:.3f} mm"
                        )
        return GeometryValidation(valid=not messages, messages=messages)

    @staticmethod
    def tolerance_stack(
        chain: list[tuple[float, float]],
    ) -> dict[str, float]:
        """Worst-case linear tolerance stack-up over a dimension chain.

        ``chain`` is a list of ``(nominal, tolerance)`` pairs; returns the
        nominal sum and worst-case (absolute) and statistical (RSS) tolerances.
        """
        if not chain:
            return {"nominal": 0.0, "worst": 0.0, "rss": 0.0}
        nominal = sum(dim for dim, _ in chain)
        worst = sum(tol for _, tol in chain)
        rss = math.sqrt(sum(tol * tol for _, tol in chain))
        return {"nominal": nominal, "worst": worst, "rss": rss}


__all__ = [
    "GeometryReasoner",
    "GeometryValidation",
    "Primitive",
]
