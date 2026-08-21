"""Tests for cadgenesis.cad.modeling (primitives, B-Rep, CSG)."""

from __future__ import annotations

import math

import pytest

from cadgenesis.cad.modeling.brep import BRepSolid
from cadgenesis.cad.modeling.csg import CSGTree
from cadgenesis.cad.modeling.primitives import (
    make_box,
    make_cone,
    make_cylinder,
    make_sphere,
    make_torus,
)


class TestPrimitives:
    def test_box_volume_surface(self) -> None:
        box = make_box(10, 5, 3)
        assert box.volume() == pytest.approx(150.0)
        assert box.surface_area() == pytest.approx(190.0)

    def test_cylinder_volume(self) -> None:
        cylinder = make_cylinder(2, 10)
        assert cylinder.volume() == pytest.approx(math.pi * 4 * 10)

    def test_sphere_volume(self) -> None:
        sphere = make_sphere(3)
        assert sphere.volume() == pytest.approx((4 / 3) * math.pi * 27)

    def test_cone_volume(self) -> None:
        cone = make_cone(3, 6)
        assert cone.volume() == pytest.approx(math.pi * 9 * 6 / 3)

    def test_torus_volume(self) -> None:
        torus = make_torus(5, 1)
        assert torus.volume() == pytest.approx(2 * math.pi**2 * 5 * 1)

    def test_aabb_centred(self) -> None:
        box = make_box(10, 4, 2)
        lo, hi = box.aabb()
        assert lo == pytest.approx((-5, -2, -1))
        assert hi == pytest.approx((5, 2, 1))

    def test_required_dimensions(self) -> None:
        with pytest.raises(TypeError):
            make_box(1, 2)  # missing height arg

    def test_dict_round_trip(self) -> None:
        box = make_box(1, 2, 3)
        restored = type(box).from_dict(box.to_dict())
        assert restored.to_dict() == box.to_dict()


class TestBRep:
    def test_prism_topology(self) -> None:
        solid = BRepSolid.from_prism(10, 5, 3)
        assert solid.validate() == []
        analysis = solid.analyze()
        assert analysis["vertices"] == 8
        assert analysis["faces"] == 6
        assert analysis["is_manifold"] is True
        assert analysis["is_closed"] is True
        assert analysis["genus"] == 0

    def test_euler_characteristic(self) -> None:
        solid = BRepSolid.from_prism(1, 1, 1)
        assert solid.analyze()["euler_characteristic"] == 2

    def test_volume_box(self) -> None:
        solid = BRepSolid.from_prism(10, 5, 3)
        assert solid.volume() == pytest.approx(150.0)

    def test_face_graph(self) -> None:
        solid = BRepSolid.from_prism(1, 1, 1)
        graph = solid.face_graph()
        assert len(graph.nodes()) == 6


class TestCSG:
    def test_tree_round_trip(self) -> None:
        tree = CSGTree()
        leaf_a = tree.new_leaf(make_box(1, 1, 1))
        leaf_b = tree.new_leaf(make_cylinder(0.5, 2))
        node = tree.new_binary("UNION", leaf_a, leaf_b)
        tree.set_root(node)
        restored = CSGTree.from_dict(tree.to_dict())
        assert restored.to_dict() == tree.to_dict()

    def test_tree_bounds(self) -> None:
        tree = CSGTree()
        leaf = tree.new_leaf(make_box(2, 2, 2))
        tree.set_root(leaf)
        lo, hi = tree.bounds()
        assert hi.x > lo.x

    def test_validate(self) -> None:
        tree = CSGTree()
        leaf = tree.new_leaf(make_box(1, 1, 1))
        tree.set_root(leaf)
        assert tree.validate() == []

    def test_validate_empty(self) -> None:
        tree = CSGTree()
        assert tree.validate() == ["CSG tree has no root"]

    def test_union_convenience(self) -> None:
        tree = CSGTree()
        a = tree.new_leaf(make_box(1, 1, 1))
        b = tree.new_leaf(make_box(1, 1, 1))
        node = tree.union(a, b)
        tree.set_root(node)
        assert node.op == "UNION"
        assert tree.history
