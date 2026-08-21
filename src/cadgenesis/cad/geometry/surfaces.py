"""cadgenesis.cad.geometry.surfaces
=================================
Surface modelling operations beyond curve/surface evaluation: trimming and
stitching of sampled surface patches.

A "surface patch" here is a rectangular grid of ``Vec`` samples produced by
:mod:`cadgenesis.cad.geometry.curves` (e.g. ``ruled_surface_points``,
``lofted_surface_points`` or ``NurbsSurface.sample_grid``).  Trim operates in
the 2D *parameter domain* of the patch (u along columns, v along rows, both in
``[0, 1]``); stitch merges two patches that share a boundary column or row.

Pure Python — no numpy dependency, matching the rest of the CAD core.
"""

from __future__ import annotations

from collections.abc import Sequence

from cadgenesis.cad.geometry.core import Vec

__all__ = [
    "SurfacePatch",
    "point_in_polygon",
    "stitch_surfaces",
    "trim_surface",
]


class SurfacePatch:
    """A rectangular surface patch sampled as a ``(rows) x (cols)`` grid.

    ``points[i][j]`` is the sample at parameter ``(v=i/(rows-1), u=j/(cols-1))``
    with ``u`` along columns and ``v`` along rows.  A cell may be ``None``
    after trimming, marking it as *removed*.
    """

    def __init__(self, points: list[list[Vec | None]]) -> None:
        if not points or not points[0]:
            raise ValueError("a surface patch needs a non-empty grid")
        width = len(points[0])
        if any(len(row) != width for row in points):
            raise ValueError("surface patch rows must have equal width")
        self.points = [list(row) for row in points]

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[Vec]]) -> SurfacePatch:
        return cls([[v for v in row] for row in grid])

    @property
    def rows(self) -> int:
        return len(self.points)

    @property
    def cols(self) -> int:
        return len(self.points[0])

    def parameter(self, i: int, j: int) -> tuple[float, float]:
        """Normalized (u, v) parameter of cell ``(i, j)``."""
        u = j / (self.cols - 1) if self.cols > 1 else 0.0
        v = i / (self.rows - 1) if self.rows > 1 else 0.0
        return u, v

    def point(self, i: int, j: int) -> Vec | None:
        return self.points[i][j]

    def is_trimmed(self) -> bool:
        return any(cell is None for row in self.points for cell in row)

    def count(self) -> int:
        return sum(cell is not None for row in self.points for cell in row)

    def bounds(self) -> tuple[Vec, Vec]:
        """AABB of the present samples (raises if everything is trimmed)."""
        present = [p for row in self.points for p in row if p is not None]
        if not present:
            raise ValueError("cannot compute bounds of an empty trimmed patch")
        xs = [p.x for p in present]
        ys = [p.y for p in present]
        zs = [p.z for p in present]
        return Vec(min(xs), min(ys), min(zs)), Vec(max(xs), max(ys), max(zs))

    def to_grid(self) -> list[list[Vec]]:
        """Return the raw grid (``None`` cells removed)."""
        return [[p for p in row if p is not None] for row in self.points]


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (boundary counts as inside)."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def trim_surface(
    patch: SurfacePatch,
    trim_loop: Sequence[tuple[float, float]],
) -> SurfacePatch:
    """Trim a patch by a closed loop in parameter space (u, v in ``[0, 1]``).

    Cells whose ``(u, v)`` parameter falls *outside* the loop are marked
    ``None`` (removed); cells inside or on the boundary are preserved.
    """
    loop = list(trim_loop)
    if len(loop) < 3:
        raise ValueError("a trim loop needs at least 3 vertices")
    if any(not (0.0 <= coord <= 1.0) for vertex in loop for coord in vertex):
        raise ValueError("trim loop coordinates must lie in [0, 1]^2")
    trimmed: list[list[Vec | None]] = []
    for i in range(patch.rows):
        row: list[Vec | None] = []
        for j in range(patch.cols):
            u, v = patch.parameter(i, j)
            if point_in_polygon((u, v), loop):
                row.append(patch.point(i, j))
            else:
                row.append(None)
        trimmed.append(row)
    return SurfacePatch(trimmed)


def _matches(a: Vec, b: Vec, tolerance: float) -> bool:
    return (a - b).norm() <= tolerance


def _require_point(point: Vec | None) -> Vec:
    if point is None:
        raise ValueError("untrimmed patch contains a missing sample")
    return point


def stitch_surfaces(
    a: SurfacePatch,
    b: SurfacePatch,
    seam_tolerance: float = 1e-6,
) -> SurfacePatch:
    """Stitch two patches that share a boundary column or row.

    The shared boundary is the last column of ``a`` against the first column
    of ``b`` (or, failing that, the last row of ``a`` against the first row of
    ``b``).  Matching boundary samples are merged; the duplicate seam is
    dropped.  Raises ``ValueError`` when no seam matches within tolerance or
    when the patches are trimmed at the seam.
    """
    if seam_tolerance < 0:
        raise ValueError("seam_tolerance must be non-negative")
    if a.rows < 2 or b.rows < 2:
        raise ValueError("stitching needs patches with at least 2 rows")

    # Try column seam: a.last-column vs b.first-column (same row count).
    if a.rows == b.rows and not a.is_trimmed() and not b.is_trimmed():
        seam = [
            _matches(
                _require_point(a.point(i, a.cols - 1)),
                _require_point(b.point(i, 0)),
                seam_tolerance,
            )
            for i in range(a.rows)
        ]
        if all(seam):
            merged: list[list[Vec | None]] = []
            for i in range(a.rows):
                row = [a.point(i, j) for j in range(a.cols - 1)]
                row.extend(b.point(i, j) for j in range(b.cols))
                merged.append(row)
            return SurfacePatch(merged)

    # Try row seam: a.last-row vs b.first-row (same column count).
    if a.cols == b.cols and not a.is_trimmed() and not b.is_trimmed():
        seam = [
            _matches(
                _require_point(a.point(a.rows - 1, j)),
                _require_point(b.point(0, j)),
                seam_tolerance,
            )
            for j in range(a.cols)
        ]
        if all(seam):
            merged = [[a.point(i, j) for j in range(a.cols)] for i in range(a.rows - 1)]
            merged.extend([b.point(i, j) for j in range(b.cols)] for i in range(b.rows))
            return SurfacePatch(merged)

    raise ValueError(
        "patches do not share a boundary within seam_tolerance; "
        "ensure matching edge samples and equal edge lengths"
    )
