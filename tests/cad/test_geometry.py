"""Tests for cadgenesis.cad.geometry (Vec, transforms, curves, NURBS)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from cadgenesis.cad.geometry.core import Axis, Frame, Plane, Transform, Vec
from cadgenesis.cad.geometry.curves import (
    NurbsCurve,
    NurbsSurface,
    bezier_curve,
    bezier_point,
    knot_vector,
)


class TestVec:
    def test_arithmetic(self) -> None:
        a = Vec(1, 2, 3)
        b = Vec(4, 5, 6)
        assert (a + b) == Vec(5, 7, 9)
        assert (b - a) == Vec(3, 3, 3)
        assert a * 2 == Vec(2, 4, 6)

    def test_dot_cross(self) -> None:
        x, y, z = Vec(1, 0, 0), Vec(0, 1, 0), Vec(0, 0, 1)
        assert x.dot(y) == 0
        assert x.cross(y) == z

    def test_length_normalize(self) -> None:
        v = Vec(3, 4, 0)
        assert v.norm() == pytest.approx(5)
        unit = v.normalized()
        assert unit.norm() == pytest.approx(1)

    def test_distance(self) -> None:
        assert Vec(0, 0, 0).distance_to(Vec(0, 0, 5)) == pytest.approx(5)

    def test_to_list_from_sequence(self) -> None:
        v = Vec.from_sequence((1, 2, 3))
        assert v.to_list() == [1, 2, 3]


class TestTransform:
    def test_identity(self) -> None:
        t = Transform.identity()
        v = t.apply(Vec(1, 2, 3))
        assert v == Vec(1, 2, 3)

    def test_translation(self) -> None:
        t = Transform.translation(10, 0, 0)
        assert t.apply(Vec(1, 2, 3)) == Vec(11, 2, 3)

    def test_scale(self) -> None:
        t = Transform.scale(2.0)
        assert t.apply(Vec(1, 2, 3)) == Vec(2, 4, 6)

    def test_compose(self) -> None:
        t1 = Transform.translation(1, 0, 0)
        t2 = Transform.translation(0, 2, 0)
        combined = t1.composed(t2)
        assert combined.apply(Vec(0, 0, 0)) == Vec(1, 2, 0)


class TestCurves:
    def test_bezier_linear(self) -> None:
        assert bezier_point([Vec(0, 0, 0), Vec(10, 0, 0)], 0.0) == Vec(0, 0, 0)
        assert bezier_point([Vec(0, 0, 0), Vec(10, 0, 0)], 1.0) == Vec(10, 0, 0)
        assert bezier_point([Vec(0, 0, 0), Vec(10, 0, 0)], 0.5).x == pytest.approx(5.0)

    def test_nurbs_quadratic(self) -> None:
        curve = NurbsCurve(2, [(0, 0, 0), (1, 1, 0), (2, 0, 0)])
        point = curve.evaluate(0.5)
        assert point[0] == pytest.approx(1.0, abs=1e-6)
        assert point[1] == pytest.approx(0.5, abs=1e-6)
        assert point[2] == pytest.approx(0.0, abs=1e-6)

    def test_bezier_curve_sampling(self) -> None:
        points = bezier_curve([Vec(0, 0, 0), Vec(10, 0, 0)], samples=8)
        assert len(points) == 8
        assert points[0] == Vec(0, 0, 0)
        assert points[-1] == Vec(10, 0, 0)

    def test_knot_vector_uniform(self) -> None:
        knots = knot_vector(2, 4, uniform=True)
        assert len(knots) == 7
        assert knots[0] == 0.0
        assert all(b >= a for a, b in pairwise(knots))

    def test_knot_vector_count(self) -> None:
        knots = knot_vector(3, 5)
        assert len(knots) == 9


class TestNurbsSurface:
    def test_sample_grid_shape(self) -> None:
        controls = [[Vec(i, j, 0) for j in range(3)] for i in range(3)]
        surface = NurbsSurface(2, 2, controls)
        grid = surface.sample_grid(4, 4)
        assert len(grid) == 4
        assert len(grid[0]) == 4
        assert grid[0][0] == controls[0][0]
        assert grid[-1][-1] == controls[-1][-1]

    def test_evaluate_midpoint(self) -> None:
        controls = [
            [Vec(0, 0, 0), Vec(1, 0, 0), Vec(2, 0, 0)],
            [Vec(0, 1, 0), Vec(1, 1, 0), Vec(2, 1, 0)],
        ]
        surface = NurbsSurface(1, 1, controls)
        point = surface.evaluate(0.5, 0.5)
        assert point.x == pytest.approx(0.5, abs=1e-6)
        assert point.y == pytest.approx(0.5, abs=1e-6)
        assert surface.evaluate(0.0, 0.0) == controls[0][0]


class TestPlane:
    def test_xy(self) -> None:
        plane = Plane.xy()
        assert plane.normal == Vec(0, 0, 1)
        assert plane.point == Vec(0, 0, 0)

    def test_point_on_plane(self) -> None:
        plane = Plane(Vec(0, 0, 0), Vec(0, 0, 1))
        assert plane.signed_distance(Vec(1, 2, 0)) == pytest.approx(0)
        assert plane.signed_distance(Vec(1, 2, 5)) == pytest.approx(5)


class TestAxis:
    def test_creation(self) -> None:
        axis = Axis(Vec(0, 0, 0), Vec(0, 0, 1))
        assert axis.direction == Vec(0, 0, 1)


class TestFrame:
    def test_identity_frame(self) -> None:
        frame = Frame(Vec(0, 0, 0), Vec(1, 0, 0), Vec(0, 1, 0), Vec(0, 0, 1))
        assert frame.origin == Vec(0, 0, 0)
        assert frame.z_axis == Vec(0, 0, 1)
