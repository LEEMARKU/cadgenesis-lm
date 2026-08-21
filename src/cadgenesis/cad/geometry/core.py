"""cadgenesis.cad.geometry.core
=============================
Foundational 3D math for the CAD Intelligence core: vectors, points,
reference frames, planes, axes and rigid transforms.

Everything here is pure Python (no numpy requirement) so the CAD kernel runs
in any environment that can install :mod:`cadgenesis`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from typing_extensions import Self

EPS = 1e-9


class Vec(tuple):
    """An immutable 3-vector with arithmetic helpers.

    Wraps a ``(x, y, z)`` tuple so vectors stay hashable (useful as dict
    keys in topology graphs) while gaining vector operations.
    """

    __slots__ = ()

    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Self:
        return tuple.__new__(cls, (float(x), float(y), float(z)))

    # -- component access ---------------------------------------------------
    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other: object) -> Vec:
        other = _coerce(other)
        return Vec(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other: object) -> Vec:
        other = _coerce(other)
        return Vec(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    def __mul__(self, scalar: float) -> Vec:  # type: ignore[override]
        return Vec(self[0] * scalar, self[1] * scalar, self[2] * scalar)

    def __rmul__(self, scalar: float) -> Vec:  # type: ignore[override]
        return Vec(self[0] * scalar, self[1] * scalar, self[2] * scalar)

    def __truediv__(self, scalar: float) -> Vec:
        return Vec(self[0] / scalar, self[1] / scalar, self[2] / scalar)

    def __neg__(self) -> Vec:
        return Vec(-self[0], -self[1], -self[2])

    def __abs__(self) -> float:
        return self.norm()

    # -- vector ops ----------------------------------------------------------
    def norm(self) -> float:
        return math.sqrt(self[0] * self[0] + self[1] * self[1] + self[2] * self[2])

    def squared_norm(self) -> float:
        return self[0] * self[0] + self[1] * self[1] + self[2] * self[2]

    def normalized(self) -> Vec:
        n = self.norm()
        if n < EPS:
            raise ZeroDivisionError("cannot normalize a zero vector")
        return self / n

    def dot(self, other: object) -> float:
        other = _coerce(other)
        return self[0] * other[0] + self[1] * other[1] + self[2] * other[2]

    def cross(self, other: object) -> Vec:
        other = _coerce(other)
        return Vec(
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        )

    def distance_to(self, other: object) -> float:
        return (self - _coerce(other)).norm()

    def almost_equal(self, other: object, tol: float = 1e-6) -> bool:
        return self.distance_to(other) <= tol

    def to_list(self) -> list[float]:
        return [self[0], self[1], self[2]]

    def to_tuple(self) -> tuple[float, float, float]:
        return (self[0], self[1], self[2])

    @staticmethod
    def from_sequence(seq: Sequence[float]) -> Vec:
        if len(seq) != 3:
            raise ValueError(f"expected 3 components, got {len(seq)}")
        return Vec(seq[0], seq[1], seq[2])


def _coerce(value: object) -> Vec:
    if isinstance(value, Vec):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"expected 3 components, got {len(value)}")
        return Vec(value[0], value[1], value[2])
    raise TypeError(f"expected a 3-vector, got {type(value).__name__}")


def lerp(a: Vec, b: Vec, t: float) -> Vec:
    """Linear interpolation between two points."""
    return a + (b - a) * t


def angle_between(a: Vec, b: Vec) -> float:
    """Angle (radians) between two vectors, always in [0, pi]."""
    denom = a.norm() * b.norm()
    if denom < EPS:
        return 0.0
    cos_angle = max(-1.0, min(1.0, a.dot(b) / denom))
    return math.acos(cos_angle)


def distance_point_line(p: Vec, a: Vec, b: Vec) -> float:
    """Distance from point ``p`` to the infinite line through ``a``, ``b``."""
    direction = b - a
    length = direction.norm()
    if length < EPS:
        return p.distance_to(a)
    return (direction.cross(p - a)).norm() / length


def project_point_line(p: Vec, a: Vec, b: Vec) -> Vec:
    """Project ``p`` onto the infinite line through ``a``, ``b``."""
    direction = b - a
    t = (p - a).dot(direction) / direction.squared_norm()
    return a + direction * t


def is_parallel(a: Vec, b: Vec, tol: float = 1e-6) -> bool:
    """True when two direction vectors are (anti)parallel."""
    return abs(a.cross(b).norm()) <= tol * max(1.0, a.norm() * b.norm())


def is_perpendicular(a: Vec, b: Vec, tol: float = 1e-6) -> bool:
    """True when two direction vectors are orthogonal."""
    return abs(a.dot(b)) <= tol * max(1.0, a.norm() * b.norm())


@dataclass(frozen=True)
class Axis:
    """An infinite axis: a point and a direction vector."""

    origin: Vec
    direction: Vec

    def __post_init__(self) -> None:
        if not isinstance(self.origin, Vec) or not isinstance(self.direction, Vec):
            object.__setattr__(self, "origin", Vec.from_sequence(self.origin))
            object.__setattr__(self, "direction", Vec.from_sequence(self.direction))
        if self.direction.norm() < EPS:
            raise ValueError("axis direction must be non-zero")


@dataclass(frozen=True)
class Plane:
    """An infinite plane defined by a point and a unit normal."""

    point: Vec
    normal: Vec

    def __post_init__(self) -> None:
        if not isinstance(self.point, Vec) or not isinstance(self.normal, Vec):
            object.__setattr__(self, "point", Vec.from_sequence(self.point))
            object.__setattr__(self, "normal", Vec.from_sequence(self.normal))
        n = self.normal.norm()
        if n < EPS:
            raise ValueError("plane normal must be non-zero")
        object.__setattr__(self, "normal", self.normal / n)

    def signed_distance(self, p: Vec) -> float:
        """Signed distance from point ``p`` to the plane."""
        return self.normal.dot(p - self.point)

    def contains(self, p: Vec, tol: float = 1e-6) -> bool:
        return abs(self.signed_distance(p)) <= tol

    def offset(self, distance: float) -> Plane:
        return Plane(self.point + self.normal * distance, self.normal)

    @staticmethod
    def xy(z: float = 0.0) -> Plane:
        return Plane(Vec(0, 0, z), Vec(0, 0, 1))

    @staticmethod
    def from_three_points(a: Vec, b: Vec, c: Vec) -> Plane:
        normal = (b - a).cross(c - a)
        return Plane(a, normal)


@dataclass(frozen=True)
class Frame:
    """An oriented 3D reference frame (right-handed orthonormal basis)."""

    origin: Vec
    x_axis: Vec
    y_axis: Vec
    z_axis: Vec

    def __post_init__(self) -> None:
        if not isinstance(self.origin, Vec):
            object.__setattr__(self, "origin", Vec.from_sequence(self.origin))
        for name in ("x_axis", "y_axis", "z_axis"):
            value = getattr(self, name)
            if not isinstance(value, Vec):
                value = Vec.from_sequence(value)
                object.__setattr__(self, name, value)
        if abs(self.z_axis.dot(self.x_axis.cross(self.y_axis)) - 1.0) > 1e-6:
            raise ValueError("frame basis is not right-handed orthonormal")

    def to_plane(self) -> Plane:
        return Plane(self.origin, self.z_axis)

    def local_to_world(self, point: Vec) -> Vec:
        return self.origin + self.x_axis * point.x + self.y_axis * point.y + self.z_axis * point.z

    @staticmethod
    def world() -> Frame:
        return Frame(Vec(0, 0, 0), Vec(1, 0, 0), Vec(0, 1, 0), Vec(0, 0, 1))

    @staticmethod
    def from_z_axis(origin: Vec, z_axis: Vec) -> Frame:
        """Build a frame whose +Z points along ``z_axis``.

        Uses an arbitrary-but-deterministic up-vector when the axis is nearly
        parallel to the world Z axis.
        """
        z = z_axis.normalized()
        reference = Vec(0, 0, 1) if abs(z.z) < 0.999 else Vec(1, 0, 0)
        x = reference.cross(z).normalized()
        y = z.cross(x)
        return Frame(Vec.from_sequence(origin), x, y, z)


class Transform:
    """Rigid body transform: rotation (angle/axis or matrix) + translation.

    Stored as a 4x4 homogeneous matrix (list of 4 rows of 4 floats).
    """

    def __init__(self, matrix: Sequence[Sequence[float]] | None = None) -> None:
        if matrix is None:
            self.matrix = _identity_matrix()
        else:
            rows = [list(row) for row in matrix]
            if len(rows) != 4 or any(len(row) != 4 for row in rows):
                raise ValueError("transform matrix must be 4x4")
            self.matrix = rows

    # -- constructors --------------------------------------------------------
    @classmethod
    def identity(cls) -> Transform:
        return cls(_identity_matrix())

    @classmethod
    def translation(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Transform:
        return cls(_translation_matrix(x, y, z))

    @classmethod
    def from_vec(cls, delta: Vec) -> Transform:
        return cls.translation(*delta.to_tuple())

    @classmethod
    def rotation(cls, angle: float, axis: Vec) -> Transform:
        """Rotation about a direction vector by ``angle`` radians."""
        n = axis.normalized()
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        x, y, z = n.x, n.y, n.z
        matrix = [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return cls(matrix)

    @classmethod
    def euler(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Transform:
        """Rotation-only transform from ZYX euler angles (radians)."""
        rx = cls.rotation(x, Vec(1, 0, 0)).matrix
        ry = cls.rotation(y, Vec(0, 1, 0)).matrix
        rz = cls.rotation(z, Vec(0, 0, 1)).matrix
        return cls(_matmul(rz, _matmul(ry, rx)))

    @classmethod
    def scale(cls, sx: float = 1.0, sy: float | None = None, sz: float | None = None) -> Transform:
        sy = sx if sy is None else sy
        sz = sx if sz is None else sz
        return cls(
            [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    # -- composition ----------------------------------------------------------
    def composed(self, other: Transform) -> Transform:
        """Matrix product ``self * other`` (apply other first)."""
        return Transform(_matmul(self.matrix, other.matrix))

    def __mul__(self, other: Transform) -> Transform:
        return self.composed(other)

    def inverted(self) -> Transform:
        # Rigid (possibly non-uniform scaled) affine inverse via adjugate.
        m = self.matrix
        a = [
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        ]
        det = _det3(a)
        if abs(det) < 1e-12:
            raise ValueError("transform is singular and cannot be inverted")
        inv = _inverse3(a)
        tx = m[0][3]
        ty = m[1][3]
        tz = m[2][3]
        out = _identity_matrix()
        for i in range(3):
            for j in range(3):
                out[i][j] = inv[i][j]
        out[0][3] = -(inv[0][0] * tx + inv[0][1] * ty + inv[0][2] * tz)
        out[1][3] = -(inv[1][0] * tx + inv[1][1] * ty + inv[1][2] * tz)
        out[2][3] = -(inv[2][0] * tx + inv[2][1] * ty + inv[2][2] * tz)
        return Transform(out)

    # -- application ----------------------------------------------------------
    def apply(self, point: Vec) -> Vec:
        m = self.matrix
        x = m[0][0] * point.x + m[0][1] * point.y + m[0][2] * point.z + m[0][3]
        y = m[1][0] * point.x + m[1][1] * point.y + m[1][2] * point.z + m[1][3]
        z = m[2][0] * point.x + m[2][1] * point.y + m[2][2] * point.z + m[2][3]
        return Vec(x, y, z)

    def apply_direction(self, direction: Vec) -> Vec:
        m = self.matrix
        return Vec(
            m[0][0] * direction.x + m[0][1] * direction.y + m[0][2] * direction.z,
            m[1][0] * direction.x + m[1][1] * direction.y + m[1][2] * direction.z,
            m[2][0] * direction.x + m[2][1] * direction.y + m[2][2] * direction.z,
        )

    def apply_many(self, points: Iterable[Vec]) -> list[Vec]:
        return [self.apply(p) for p in points]

    def to_list(self) -> list[list[float]]:
        return [list(row) for row in self.matrix]

    def to_frame(self) -> Frame:
        m = self.matrix
        return Frame(
            Vec(m[0][3], m[1][3], m[2][3]),
            Vec(m[0][0], m[1][0], m[2][0]),
            Vec(m[0][1], m[1][1], m[2][1]),
            Vec(m[0][2], m[1][2], m[2][2]),
        )


# ---------------------------------------------------------------------------
# internal matrix helpers
# ---------------------------------------------------------------------------


def _identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation_matrix(x: float, y: float, z: float) -> list[list[float]]:
    m = _identity_matrix()
    m[0][3] = float(x)
    m[1][3] = float(y)
    m[2][3] = float(z)
    return m


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j]
            for j in range(4)
        ]
        for i in range(4)
    ]


def _det3(a: list[list[float]]) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _inverse3(a: list[list[float]]) -> list[list[float]]:
    det = _det3(a)
    inv = [
        [
            a[1][1] * a[2][2] - a[1][2] * a[2][1],
            a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][1] * a[1][2] - a[0][2] * a[1][1],
        ],
        [
            a[1][2] * a[2][0] - a[1][0] * a[2][2],
            a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][2] * a[1][0] - a[0][0] * a[1][2],
        ],
        [
            a[1][0] * a[2][1] - a[1][1] * a[2][0],
            a[0][1] * a[2][0] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[0][1] * a[1][0],
        ],
    ]
    for row in inv:
        for i in range(3):
            row[i] /= det
    return inv


__all__ = [
    "EPS",
    "Axis",
    "Frame",
    "Plane",
    "Transform",
    "Vec",
    "angle_between",
    "distance_point_line",
    "is_parallel",
    "is_perpendicular",
    "lerp",
    "project_point_line",
]
