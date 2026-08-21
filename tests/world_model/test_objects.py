"""tests/world_model/test_objects.py
====================================
Unit tests for the Pillar-4 world-model object representation.
"""

from __future__ import annotations

import math

import pytest

from cadgenesis.world_model import (
    STOCK_MATERIALS,
    Material,
    ObjectGraph,
    WorldObject,
    make_object,
)


class TestWorldObject:
    def test_invalid_feature(self):
        with pytest.raises(ValueError):
            WorldObject(name="x", feature="shaft")

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            WorldObject(name="x", confidence=1.5)

    def test_block_bounds_and_volume(self):
        obj = make_object("block", "plate", {"length": 100, "width": 60, "height": 8})
        lo, hi = obj.bounds()
        assert hi.x - lo.x == pytest.approx(100.0)
        assert hi.y - lo.y == pytest.approx(60.0)
        assert hi.z - lo.z == pytest.approx(8.0)
        assert obj.volume_estimate() == pytest.approx(48000.0)

    def test_cylinder_volume(self):
        obj = make_object("cylinder", "shaft", {"radius": 10, "height": 80})
        assert obj.volume_estimate() == pytest.approx(math.pi * 100 * 80)

    def test_mass_uses_material(self):
        obj = make_object(
            "block", "plate", {"length": 100, "width": 100, "height": 10}, material="steel"
        )
        # 100000 mm^3 * 7850 kg/m^3 / 1e9 = 0.785 kg
        assert obj.mass() == pytest.approx(0.785, abs=1e-3)

    def test_no_material_mass_zero(self):
        obj = make_object("block", "plate")
        assert obj.mass() == 0.0

    def test_roundtrip_serialization(self):
        obj = make_object("cylinder", "shaft", {"radius": 3, "height": 50}, material="titanium")
        restored = WorldObject.from_dict(obj.to_dict())
        assert restored.name == obj.name
        assert restored.feature == obj.feature
        assert restored.parameters == obj.parameters
        assert restored.material is not None
        assert restored.material.name == "titanium"
        assert restored.object_id == obj.object_id
        assert restored.position == obj.position

    def test_unique_ids(self):
        a = make_object("block", "a")
        b = make_object("block", "b")
        assert a.object_id != b.object_id


class TestMaterial:
    def test_stock_materials(self):
        assert "steel" in STOCK_MATERIALS
        assert STOCK_MATERIALS["steel"].yield_strength_mpa == 250.0

    def test_material_roundtrip(self):
        m = Material.from_dict(Material("aluminum", 2700.0, 95.0).to_dict())
        assert m.name == "aluminum"


class TestObjectGraph:
    def test_relate_children_roots(self):
        graph = ObjectGraph()
        parent = make_object("block", "base")
        child = make_object("block", "top")
        graph.add(parent)
        graph.add(child)
        graph.relate(parent.object_id, child.object_id, "mounts")
        assert graph.roots() == [parent]
        assert [c.object_id for c in graph.children(parent.object_id)] == [child.object_id]
        assert graph.root_for(child) is parent

    def test_relate_duplicate_parent_raises(self):
        graph = ObjectGraph()
        a = graph.add(make_object("block", "a"))
        b = graph.add(make_object("block", "b"))
        c = graph.add(make_object("block", "c"))
        graph.relate(a.object_id, c.object_id)
        with pytest.raises(ValueError):
            graph.relate(b.object_id, c.object_id)

    def test_set_pose_from_dict(self):
        graph = ObjectGraph()
        obj = graph.add(make_object("block", "a"))
        graph.set_pose(
            obj.object_id,
            {"matrix": [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        )
        assert obj.position.x == pytest.approx(10.0)
