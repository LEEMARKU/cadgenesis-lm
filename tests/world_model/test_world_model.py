"""tests/world_model/test_world_model.py
========================================
Unit tests for the Pillar-4 facade, integration and metrics.
"""

from __future__ import annotations

from cadgenesis.cad.geometry.core import Transform
from cadgenesis.world_model import (
    WorldModelIntegration,
    WorldModelSystem,
)
from cadgenesis.world_model.objects import BoundaryCondition, LoadCase


class TestWorldModelSystem:
    def test_capabilities_dispatched(self):
        wm = WorldModelSystem(name="box")
        plate = wm.add_object(
            "block", "base", {"length": 100, "width": 60, "height": 8}, material="steel"
        )
        bracket = wm.add_object(
            "block",
            "bracket",
            {"length": 40, "width": 30, "height": 50},
            pose=Transform.translation(0, 0, 60),
        )
        clearance = wm.reason("clearance", a=plate, b=bracket, minimum=5, axis="z")
        assert clearance.passed
        assert not wm.reason("overlap", a=plate, b=bracket)
        assert wm.reason("distance", a=plate, b=bracket) == 60.0

    def test_unknown_capability(self):
        wm = WorldModelSystem()
        try:
            wm.reason("nonsense")
        except KeyError:
            return
        raise AssertionError("expected KeyError for unknown capability")

    def test_snapshot_restore(self):
        wm = WorldModelSystem(name="a")
        wm.add_object("block", "part", {"length": 10, "width": 10, "height": 10}, material="steel")
        state = wm.snapshot()
        wm2 = WorldModelSystem(name="b")
        wm2.restore(state)
        assert wm2.graph.objects[0].name == "part"
        assert wm2.graph.objects[0].mass() > 0

    def test_safety_reasoning(self):
        wm = WorldModelSystem()
        beam = wm.add_object(
            "block", "beam", {"length": 200, "width": 20, "height": 20}, material="steel"
        )
        load = LoadCase("static", [BoundaryCondition(kind="force", magnitude=2000.0)])
        result = wm.reason("safety", object=beam, load_case=load)
        assert result.passed


class TestWorldModelIntegration:
    def test_cad_document_projection(self):
        wm = WorldModelSystem()
        wm.add_object("block", "plate", {"length": 100, "width": 60, "height": 8}, material="steel")
        wm.add_object("cylinder", "shaft", {"radius": 3, "height": 50}, material="steel")
        doc = WorldModelIntegration().to_cad_document(wm.graph)
        assert doc.feature_count == 2
        assert doc.materials == ["steel", "steel"]

    def test_multimodal_sample(self):
        wm = WorldModelSystem()
        wm.add_object("block", "plate", {"length": 100, "width": 60, "height": 8}, material="steel")
        sample = WorldModelIntegration().to_multimodal_sample(wm.graph, label="part")
        assert sample.label == "part"
        assert len(sample.inputs) == 3

    def test_memory_store_retrieve(self):
        from cadgenesis.memory import MemorySystem

        wm = WorldModelSystem()
        wm.add_object("block", "plate", {"length": 100, "width": 60, "height": 8}, material="steel")
        memory = MemorySystem()
        integration = WorldModelIntegration()
        integration.store(wm.graph, memory, key="snap")
        hits = integration.retrieve("plate", memory, top_k=2)
        assert len(hits) >= 1
