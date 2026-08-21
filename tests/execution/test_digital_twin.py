"""Tests for the digital-twin package."""

from __future__ import annotations

import pytest

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.digital_twin import DigitalTwinSystem, TwinRecord


@pytest.fixture
def twin_system() -> DigitalTwinSystem:
    return DigitalTwinSystem()


class TestMaterialize:
    def test_register_mesh(self, twin_system: DigitalTwinSystem) -> None:
        record = twin_system.register("p1", Mesh.box())
        assert isinstance(record, TwinRecord)
        assert twin_system.twin("p1") is record

    def test_register_mesh_dict(self, twin_system: DigitalTwinSystem) -> None:
        twin_system.register("p1", Mesh.box().to_dict())
        assert twin_system.twin("p1").mesh.face_count == 12

    def test_materialize_design(self, twin_system: DigitalTwinSystem) -> None:
        record = twin_system.materialize(
            "p1", {"mesh": Mesh.box(), "name": "plate", "material": "steel"}
        )
        assert record.properties["material"] == "steel"
        assert record.sync_status == "materialized"

    def test_materialize_without_mesh_raises(self, twin_system: DigitalTwinSystem) -> None:
        with pytest.raises(ValueError):
            twin_system.materialize("p1", {"name": "plate"})

    def test_unknown_twin(self, twin_system: DigitalTwinSystem) -> None:
        assert twin_system.twin("nope") is None
        assert twin_system.twins() == []


class TestSynchronize:
    def test_no_drift(self, twin_system: DigitalTwinSystem) -> None:
        twin_system.materialize("p1", {"mesh": Mesh.box(), "name": "p1"})
        status = twin_system.synchronize("p1")
        assert status["status"] == "synchronized"
        assert status["drift"] is False

    def test_sensor_drift(self, twin_system: DigitalTwinSystem) -> None:
        twin_system.materialize("p1", {"mesh": Mesh.box(), "name": "p1"})
        status = twin_system.synchronize("p1", {"volume_mm3": 1000.0})
        assert status["drift"] is True
        assert status["drift_ratio"] > 0.05

    def test_unknown_twin_status(self, twin_system: DigitalTwinSystem) -> None:
        assert twin_system.synchronize("nope")["status"] == "unknown"


class TestSimulateAndSnapshot:
    def test_simulate_caches(self, twin_system: DigitalTwinSystem) -> None:
        twin_system.materialize("p1", {"mesh": Mesh.box(), "name": "p1"})
        summary = twin_system.simulate(
            "p1",
            "structural",
            part={
                "material": {"name": "steel", "yield_strength_pa": 250e6},
                "load": {"force_n": 1000.0, "area_m2": 1e-3},
            },
        )
        assert summary["analysis_type"] == "structural"
        record = twin_system.twin("p1")
        assert record.simulations["structural"] == summary

    def test_simulate_unknown_twin(self, twin_system: DigitalTwinSystem) -> None:
        with pytest.raises(KeyError):
            twin_system.simulate("nope", "structural")

    def test_snapshot_roundtrip(self, twin_system: DigitalTwinSystem) -> None:
        twin_system.materialize("p1", {"mesh": Mesh.box(), "name": "p1"})
        snapshot = twin_system.snapshot("p1")
        assert snapshot["part_key"] == "p1"
        assert snapshot["mesh"]["vertices"]
        restored = DigitalTwinSystem().register("p1", snapshot["mesh"])
        assert restored.mesh.is_watertight()

    def test_snapshot_unknown_twin(self, twin_system: DigitalTwinSystem) -> None:
        with pytest.raises(KeyError):
            twin_system.snapshot("nope")


class TestStatus:
    def test_fleet_status(self, twin_system: DigitalTwinSystem) -> None:
        twin_system.materialize("a", {"mesh": Mesh.box(), "name": "a"})
        twin_system.materialize("b", {"mesh": Mesh.box(), "name": "b"})
        twin_system.synchronize("a")
        status = twin_system.status()
        assert status["twins"] == 2
        assert status["synced"] == 1
        assert status["drift"] == 0
        assert status["keys"] == ["a", "b"]


class TestMemoryPersistence:
    def test_memory_remember_called(self) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.entries: list[tuple[str, str, dict]] = []

            def remember(self, pool: str, key: str, content: dict) -> None:
                self.entries.append((pool, key, content))

        memory = FakeMemory()
        system = DigitalTwinSystem(memory=memory)
        system.materialize("p1", {"mesh": Mesh.box(), "name": "p1"})
        assert any(key == "twin:p1" for _, key, _ in memory.entries)

    def test_memory_failure_ignored(self) -> None:
        class BrokenMemory:
            def remember(self, pool: str, key: str, content: dict) -> None:
                raise AttributeError("no such pool")

        system = DigitalTwinSystem(memory=BrokenMemory())
        record = system.materialize("p1", {"mesh": Mesh.box(), "name": "p1"})
        assert record.part_key == "p1"
