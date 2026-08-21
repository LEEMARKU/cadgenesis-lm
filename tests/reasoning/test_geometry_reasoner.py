"""tests/reasoning/test_geometry_reasoner.py
===========================================
Unit tests for cadgenesis.reasoning.geometry_reasoner.
"""

from __future__ import annotations

import math

import pytest

from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner, Primitive


class TestPrimitive:
    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            Primitive(kind="dodecahedron", dims={})

    def test_invalid_dim_type(self):
        with pytest.raises(TypeError):
            Primitive(kind="box", dims={"length": "ten", "width": 1.0, "height": 1.0})

    def test_bad_position(self):
        with pytest.raises(ValueError):
            Primitive(kind="box", dims={"length": 1, "width": 1, "height": 1}, position=(1.0, 2.0))


class TestVolumes:
    def test_box(self):
        prim = Primitive("box", {"length": 10, "width": 4, "height": 3})
        assert GeometryReasoner.volume(prim) == pytest.approx(120.0)

    def test_cylinder(self):
        prim = Primitive("cylinder", {"radius": 2, "height": 5})
        assert GeometryReasoner.volume(prim) == pytest.approx(math.pi * 4 * 5)

    def test_sphere(self):
        prim = Primitive("sphere", {"radius": 3})
        assert GeometryReasoner.volume(prim) == pytest.approx((4 / 3) * math.pi * 27)

    def test_invalid_volume_raises(self):
        prim = Primitive("box", {"length": 10, "width": -1, "height": 3})
        with pytest.raises(ValueError):
            GeometryReasoner.volume(prim)


class TestAabb:
    def test_box_centered(self):
        prim = Primitive("box", {"length": 10, "width": 4, "height": 2})
        lo, hi = GeometryReasoner.aabb(prim)
        assert lo == (-5.0, -2.0, -1.0)
        assert hi == (5.0, 2.0, 1.0)

    def test_box_offset(self):
        prim = Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(10, 0, 0))
        lo, hi = GeometryReasoner.aabb(prim)
        assert lo[0] == 9.0
        assert hi[0] == 11.0


class TestPredicates:
    def test_overlap(self):
        a = Primitive("box", {"length": 4, "width": 4, "height": 4})
        b = Primitive("box", {"length": 4, "width": 4, "height": 4}, position=(2, 0, 0))
        assert GeometryReasoner.overlaps(a, b)

    def test_no_overlap(self):
        a = Primitive("box", {"length": 2, "width": 2, "height": 2})
        b = Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(5, 0, 0))
        assert not GeometryReasoner.overlaps(a, b)

    def test_clearance(self):
        a = Primitive("box", {"length": 2, "width": 2, "height": 2})
        b = Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(4, 0, 0))
        assert GeometryReasoner.clearance(a, b) == pytest.approx(2.0)
        assert GeometryReasoner.check_clearance(a, b, gap=1.5)

    def test_contains(self):
        inner = Primitive("box", {"length": 2, "width": 2, "height": 2})
        outer = Primitive("box", {"length": 10, "width": 10, "height": 10})
        assert GeometryReasoner.contains(inner, outer)
        assert GeometryReasoner.check_fit(inner, outer)

    def test_contains_negative(self):
        inner = Primitive("box", {"length": 12, "width": 2, "height": 2})
        outer = Primitive("box", {"length": 10, "width": 10, "height": 10})
        assert not GeometryReasoner.contains(inner, outer)

    def test_combined_bounds_empty(self):
        assert GeometryReasoner.combined_bounds([]) is None

    def test_combined_bounds(self):
        a = Primitive("box", {"length": 2, "width": 2, "height": 2})
        b = Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(5, 0, 0))
        lo, hi = GeometryReasoner.combined_bounds([a, b])
        assert lo == (-1.0, -1.0, -1.0)
        assert hi == (6.0, 1.0, 1.0)


class TestValidation:
    def test_missing_dim(self):
        prim = Primitive("box", {"length": 10, "width": 4})
        check = GeometryReasoner.validate(prim)
        assert not check.valid
        assert any("height" in m for m in check.messages)

    def test_non_positive_dim(self):
        prim = Primitive("cylinder", {"radius": 0, "height": 5})
        check = GeometryReasoner.validate(prim)
        assert not check.valid

    def test_valid(self):
        prim = Primitive("sphere", {"radius": 2})
        assert GeometryReasoner.validate(prim).valid
