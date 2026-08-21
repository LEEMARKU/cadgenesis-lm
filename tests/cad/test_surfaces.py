"""Tests for cadgenesis.cad.geometry.surfaces (trim, stitch, SurfacePatch)."""

from __future__ import annotations

import pytest

from cadgenesis.cad.geometry.core import Vec
from cadgenesis.cad.geometry.surfaces import (
    SurfacePatch,
    point_in_polygon,
    stitch_surfaces,
    trim_surface,
)


def make_patch(rows: int = 3, cols: int = 3) -> SurfacePatch:
    points = [[Vec(float(r), float(c), 0.0) for c in range(cols)] for r in range(rows)]
    return SurfacePatch.from_grid(points)


class TestSurfacePatch:
    def test_from_grid_size(self) -> None:
        patch = make_patch(3, 4)
        assert patch.rows == 3
        assert patch.cols == 4
        assert patch.count() == 12

    def test_point_roundtrip(self) -> None:
        patch = make_patch()
        p = patch.point(1, 2)
        assert (p.x, p.y) == (1.0, 2.0)

    def test_parameter_bounds(self) -> None:
        patch = make_patch(3, 3)
        u, v = patch.parameter(0, 0)
        assert (u, v) == (0.0, 0.0)
        u2, v2 = patch.parameter(2, 2)
        assert (u2, v2) == (1.0, 1.0)

    def test_is_trimmed_false_by_default(self) -> None:
        assert not make_patch().is_trimmed()

    def test_to_grid_roundtrip(self) -> None:
        patch = make_patch(2, 2)
        grid = patch.to_grid()
        assert len(grid) == 2
        assert len(grid[0]) == 2

    def test_bounds(self) -> None:
        patch = make_patch(3, 3)
        low, high = patch.bounds()
        assert low.x == 0.0
        assert high.y == 2.0


class TestPointInPolygon:
    def test_inside(self) -> None:
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert point_in_polygon((0.5, 0.5), polygon)

    def test_outside(self) -> None:
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert not point_in_polygon((2.0, 2.0), polygon)

    def test_below_edge_is_outside(self) -> None:
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert not point_in_polygon((0.5, -0.5), polygon)


class TestTrim:
    def test_trim_removes_outside_points(self) -> None:
        patch = make_patch(5, 5)
        polygon = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
        trimmed = trim_surface(patch, polygon)
        assert trimmed.is_trimmed()
        assert trimmed.count() < patch.count()
        kept = [p for row in trimmed.to_grid() for p in row if p is not None]
        assert all(1.0 <= p.x <= 3.0 and 1.0 <= p.y <= 3.0 for p in kept)

    def test_trim_whole_domain_keeps_most(self) -> None:
        patch = make_patch(5, 5)
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        trimmed = trim_surface(patch, polygon)
        assert trimmed.count() > 0

    def test_trim_no_overlap_keeps_none(self) -> None:
        patch = make_patch(5, 5)
        polygon = [(0.5, 0.9), (0.6, 0.9), (0.6, 0.95), (0.5, 0.95)]
        trimmed = trim_surface(patch, polygon)
        assert trimmed.count() == 0


class TestStitch:
    def test_stitch_side_by_side_drops_shared_seam(self) -> None:
        left = make_patch(3, 3)
        right_grid = [[Vec(float(r), float(c) + 2.0, 0.0) for c in range(3)] for r in range(3)]
        right = SurfacePatch.from_grid(right_grid)
        stitched = stitch_surfaces(left, right)
        assert stitched.rows == 3
        assert stitched.cols == 5

    def test_stitch_row_wise(self) -> None:
        top = make_patch(3, 3)
        bottom_grid = [[Vec(float(r) + 2.0, float(c), 0.0) for c in range(3)] for r in range(3)]
        bottom = SurfacePatch.from_grid(bottom_grid)
        stitched = stitch_surfaces(top, bottom)
        assert stitched.rows == 5
        assert stitched.cols == 3

    def test_stitch_requires_matching_seam(self) -> None:
        a = make_patch(3, 3)
        b = make_patch(4, 4)
        with pytest.raises(ValueError):
            stitch_surfaces(a, b)
