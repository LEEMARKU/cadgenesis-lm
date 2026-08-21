"""cadgenesis.cad.geometry.curves
===============================
Curve and surface mathematics: Bezier and NURBS evaluation, plus the
lofted / ruled surface builder used by surface modelling.

All evaluators are deterministic, pure-Python implementations of the
standard de Casteljau and Cox-de Boor algorithms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from cadgenesis.cad.geometry.core import Vec

# ---------------------------------------------------------------------------
# Bezier
# ---------------------------------------------------------------------------


def bernstein(i: int, n: int, t: float) -> float:
    """Bernstein basis polynomial B(i, n, t)."""
    return math.comb(n, i) * (t**i) * ((1.0 - t) ** (n - i))


def bezier_point(control: Sequence[Vec], t: float) -> Vec:
    """Evaluate a Bezier curve at parameter ``t`` (de Casteljau)."""
    points = [Vec.from_sequence(p) for p in control]
    if not points:
        raise ValueError("a Bezier curve needs at least one control point")
    t = max(0.0, min(1.0, float(t)))
    n = len(points)
    if n == 1:
        return points[0]
    for level in range(1, n):
        next_points = [points[i] + (points[i + 1] - points[i]) * t for i in range(n - level)]
        points = next_points
    return points[0]


def bezier_curve(control: Sequence[Vec], samples: int = 32) -> list[Vec]:
    """Sample a Bezier curve into ``samples`` points (inclusive of ends)."""
    if samples < 2:
        raise ValueError("samples must be >= 2")
    return [bezier_point(control, i / (samples - 1)) for i in range(samples)]


def bezier_arc(
    center: Vec, radius: float, start: float, end: float, samples: int = 24
) -> list[Vec]:
    """Sample a circular arc (in plane Z=center.z) as a polyline."""
    step = (end - start) / max(1, samples - 1)
    return [
        Vec(
            center.x + radius * math.cos(start + step * i),
            center.y + radius * math.sin(start + step * i),
            center.z,
        )
        for i in range(samples)
    ]


@dataclass
class BezierSurface:
    """A tensor-product Bezier surface patch."""

    control: list[list[Vec]]  # rows: (degree_u+1) x (degree_v+1)

    def __post_init__(self) -> None:
        if not self.control or not self.control[0]:
            raise ValueError("a Bezier surface needs a non-empty control grid")
        width = len(self.control[0])
        if any(len(row) != width for row in self.control):
            raise ValueError("control grid rows must have equal width")

    @property
    def degree_u(self) -> int:
        return len(self.control) - 1

    @property
    def degree_v(self) -> int:
        return len(self.control[0]) - 1

    def evaluate(self, u: float, v: float) -> Vec:
        """Evaluate the patch at parameters (u, v) in [0, 1]^2."""
        u = max(0.0, min(1.0, float(u)))
        v = max(0.0, min(1.0, float(v)))
        nu = len(self.control)
        nv = len(self.control[0])
        bu = [bernstein(i, nu - 1, u) for i in range(nu)]
        bv = [bernstein(j, nv - 1, v) for j in range(nv)]
        total = Vec(0, 0, 0)
        for i in range(nu):
            for j in range(nv):
                total = total + self.control[i][j] * (bu[i] * bv[j])
        return total

    def sample_grid(self, nu: int = 16, nv: int = 16) -> list[list[Vec]]:
        return [[self.evaluate(i / (nu - 1), j / (nv - 1)) for j in range(nv)] for i in range(nu)]


# ---------------------------------------------------------------------------
# NURBS
# ---------------------------------------------------------------------------


def knot_vector(degree: int, control_count: int, uniform: bool = False) -> list[float]:
    """Build a clamped (or uniform) knot vector for a NURBS curve.

    Clamped vectors repeat the end knots ``degree + 1`` times (open-curve
    convention, as used by CAD kernels).  A clamped vector has exactly
    ``control_count + degree + 1`` knots.
    """
    if control_count < degree + 1:
        raise ValueError("control_count must be at least degree + 1")
    if uniform:
        n = degree + control_count + 1
        return [float(i) for i in range(n)]
    last = control_count - degree
    interior = [float(v) for v in range(1, last)]
    return [0.0] * (degree + 1) + interior + [float(last)] * (degree + 1)


def _find_span(degree: int, knots: Sequence[float], u: float) -> int:
    n = len(knots) - degree - 1
    if u >= knots[n]:
        return n - 1
    if u <= knots[degree]:
        return degree
    lo, hi = degree, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def _basis_functions(degree: int, knots: Sequence[float], span: int, u: float) -> list[float]:
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    basis = [0.0] * (degree + 1)
    basis[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if denom == 0.0:
                basis[r] = 0.0
                continue
            temp = basis[r] / denom
            basis[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        basis[j] = saved
    return basis


@dataclass
class NurbsCurve:
    """A NURBS curve: degree, control points, (optional) weights, knots."""

    degree: int
    control_points: list[Vec]
    weights: list[float] | None = None
    knots: list[float] | None = None

    def __post_init__(self) -> None:
        self.control_points = [Vec.from_sequence(p) for p in self.control_points]
        if self.weights is None:
            self.weights = [1.0] * len(self.control_points)
        if len(self.weights) != len(self.control_points):
            raise ValueError("weights and control_points must have equal length")
        if self.knots is None:
            self.knots = knot_vector(self.degree, len(self.control_points))
        expected = len(self.control_points) + self.degree + 1
        if len(self.knots) != expected:
            raise ValueError(f"knot vector length {len(self.knots)} != expected {expected}")

    def evaluate(self, u: float) -> Vec:
        """Evaluate the curve at parameter ``u`` in the knot domain."""
        u = float(u)
        knots = self.knots or []
        lo, hi = knots[0], knots[-1]
        u = max(lo, min(hi, u))
        if hi > lo and u >= hi and knots[-self.degree - 1] == hi:
            # clamped curve: interpolates its last control point at u == hi
            return Vec.from_sequence(self.control_points[-1])
        if u <= lo and knots[self.degree] == lo:
            return Vec.from_sequence(self.control_points[0])
        span = _find_span(self.degree, knots, u)
        basis = _basis_functions(self.degree, knots, span, u)
        weights = self.weights or [1.0] * len(self.control_points)
        total = Vec(0, 0, 0)
        denom = 0.0
        for j in range(self.degree + 1):
            weight = weights[span - self.degree + j]
            point = self.control_points[span - self.degree + j]
            total = total + point * (basis[j] * weight)
            denom += basis[j] * weight
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("degenerate NURBS denominator")
        return total / denom

    def sample(self, samples: int = 32) -> list[Vec]:
        knots = self.knots or []
        lo, hi = knots[0], knots[-1]
        return [self.evaluate(lo + (hi - lo) * i / max(1, samples - 1)) for i in range(samples)]


@dataclass
class NurbsSurface:
    """A tensor-product NURBS surface."""

    degree_u: int
    degree_v: int
    control_points: list[list[Vec]]
    weights: list[list[float]] | None = None
    knots_u: list[float] | None = None
    knots_v: list[float] | None = None

    def __post_init__(self) -> None:
        rows = len(self.control_points)
        cols = len(self.control_points[0])
        if any(len(row) != cols for row in self.control_points):
            raise ValueError("control grid rows must have equal width")
        if self.weights is None:
            self.weights = [[1.0] * cols for _ in range(rows)]
        if self.knots_u is None:
            self.knots_u = knot_vector(self.degree_u, rows)
        if self.knots_v is None:
            self.knots_v = knot_vector(self.degree_v, cols)

    def evaluate(self, u: float, v: float) -> Vec:
        ku = self.knots_u or []
        kv = self.knots_v or []
        weights = self.weights or []
        su = _find_span(self.degree_u, ku, u)
        sv = _find_span(self.degree_v, kv, v)
        bu = _basis_functions(self.degree_u, ku, su, u)
        bv = _basis_functions(self.degree_v, kv, sv, v)
        total = Vec(0, 0, 0)
        denom = 0.0
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                w = weights[su - self.degree_u + i][sv - self.degree_v + j]
                p = self.control_points[su - self.degree_u + i][sv - self.degree_v + j]
                total = total + p * (bu[i] * bv[j] * w)
                denom += bu[i] * bv[j] * w
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("degenerate NURBS surface denominator")
        return total / denom

    def sample_grid(self, nu: int = 12, nv: int = 12) -> list[list[Vec]]:
        return [[self.evaluate(i / (nu - 1), j / (nv - 1)) for j in range(nv)] for i in range(nu)]


# ---------------------------------------------------------------------------
# Ruled / lofted surfaces
# ---------------------------------------------------------------------------


def ruled_surface_points(
    profile_a: Sequence[Vec],
    profile_b: Sequence[Vec],
    samples: int = 12,
) -> list[list[Vec]]:
    """Sample a ruled surface between two polylines of equal length."""
    a = list(profile_a)
    b = list(profile_b)
    if len(a) != len(b):
        raise ValueError("ruled surface profiles must have equal point counts")
    if len(a) < 2:
        raise ValueError("ruled surface needs at least 2 points per profile")
    return [
        [a[i] + (b[i] - a[i]) * t for i in range(len(a))]
        for t in (j / (samples - 1) for j in range(samples))
    ]


def lofted_surface_points(profiles: Sequence[Sequence[Vec]], samples: int = 12) -> list[list[Vec]]:
    """Sample a lofted (skinned) surface through >= 2 profiles.

    Profiles are connected in order; the surface interpolates between
    successive profile polylines using ruled interpolation (a standard
    lofting approximation for parallel-sectioned parts).
    """
    profile_list = [list(p) for p in profiles]
    if len(profile_list) < 2:
        raise ValueError("lofting requires at least 2 profiles")
    count = len(profile_list[0])
    if any(len(p) != count for p in profile_list):
        raise ValueError("loft profiles must have equal point counts")
    rows: list[list[Vec]] = []
    for i in range(len(profile_list) - 1):
        rows.extend(ruled_surface_points(profile_list[i], profile_list[i + 1], samples))
    return rows


__all__ = [
    "BezierSurface",
    "NurbsCurve",
    "NurbsSurface",
    "bernstein",
    "bezier_arc",
    "bezier_curve",
    "bezier_point",
    "knot_vector",
    "lofted_surface_points",
    "ruled_surface_points",
]
