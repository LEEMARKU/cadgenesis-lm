"""cadgenesis.digital_twin.twin
==============================
Digital twin system: materialization, synchronization, simulation and
memory-backed snapshots.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.modeling.brep import BRepSolid
from cadgenesis.execution.simulation import SimulationEngine


@dataclass
class TwinRecord:
    """A materialized twin of a part."""

    part_key: str
    mesh: Mesh
    brep: BRepSolid | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    simulations: dict[str, Any] = field(default_factory=dict)
    sync_status: str = "materialized"

    def summary(self) -> dict[str, Any]:
        return {
            "part_key": self.part_key,
            "sync_status": self.sync_status,
            "vertices": self.mesh.vertex_count,
            "faces": self.mesh.face_count,
            "watertight": self.mesh.is_watertight(),
            "volume_mm3": round(self.mesh.volume(), 6),
            "surface_area_mm2": round(self.mesh.surface_area(), 6),
            "simulations": sorted(self.simulations),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_key": self.part_key,
            "mesh": self.mesh.to_dict(),
            "brep": self.brep.to_dict() if self.brep is not None else None,
            "properties": dict(self.properties),
            "simulations": dict(self.simulations),
            "sync_status": self.sync_status,
        }


class DigitalTwinSystem:
    """Materialize, synchronize and simulate part twins."""

    def __init__(
        self,
        simulation: SimulationEngine | None = None,
        memory: Any = None,
    ) -> None:
        self.simulation = simulation or SimulationEngine()
        self.memory = memory
        self._twins: dict[str, TwinRecord] = {}

    def register(self, part_key: str, mesh: Mesh | dict[str, Any]) -> TwinRecord:
        """Register a twin for ``part_key`` from a mesh (or mesh dict)."""
        solid = mesh if isinstance(mesh, Mesh) else Mesh.from_dict(mesh)
        record = TwinRecord(part_key=part_key, mesh=solid)
        self._twins[part_key] = record
        return record

    def materialize(
        self,
        part_key: str,
        design: dict[str, Any],
    ) -> TwinRecord:
        """Materialize a twin from a design dict with a mesh and metadata."""
        mesh = design.get("mesh")
        if not isinstance(mesh, Mesh):
            raise ValueError("design must carry a Mesh under 'mesh'")
        record = TwinRecord(
            part_key=part_key,
            mesh=mesh,
            properties={
                "material": design.get("material"),
                "processes": design.get("processes"),
                "source": design.get("name", part_key),
            },
        )
        self._twins[part_key] = record
        self._persist(record)
        return record

    def twin(self, part_key: str) -> TwinRecord | None:
        """Look up a registered twin."""
        return self._twins.get(part_key)

    def twins(self) -> list[str]:
        return sorted(self._twins)

    def synchronize(
        self,
        part_key: str,
        sensor_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a twin with execution results / sensor data; reports drift."""
        record = self._twins.get(part_key)
        if record is None:
            return {"status": "unknown", "drift": False}
        volume = record.mesh.volume()
        expected = float(record.properties.get("nominal_volume_mm3") or volume)
        drift = abs(volume - expected) / max(expected, 1e-9)
        if sensor_data:
            record.properties["last_sensor_read"] = dict(sensor_data)
            measured = float(sensor_data.get("volume_mm3") or volume)
            drift = abs(measured - expected) / max(expected, 1e-9)
        record.sync_status = "synchronized"
        if drift > 0.05:
            record.sync_status = "drift"
        self._persist(record)
        return {
            "status": record.sync_status,
            "drift": drift > 0.05,
            "drift_ratio": round(drift, 6),
        }

    def simulate(self, part_key: str, analysis_type: str, **params: Any) -> dict[str, Any]:
        """Run an analytic simulation against the twin and cache the result."""
        record = self._twins.get(part_key)
        if record is None:
            raise KeyError(f"unknown twin {part_key!r}")
        result = self.simulation.run(analysis_type, **params)
        record.simulations[analysis_type] = result.summary()
        return record.simulations[analysis_type]

    def snapshot(self, part_key: str) -> dict[str, Any]:
        """Serializable twin snapshot (also persisted to memory)."""
        record = self._twins.get(part_key)
        if record is None:
            raise KeyError(f"unknown twin {part_key!r}")
        snapshot = record.to_dict()
        self._persist(record)
        return snapshot

    def status(self) -> dict[str, Any]:
        """Fleet status across registered twins."""
        records = list(self._twins.values())
        return {
            "twins": len(records),
            "keys": sorted(self._twins),
            "synced": sum(1 for r in records if r.sync_status == "synchronized"),
            "drift": sum(1 for r in records if r.sync_status == "drift"),
            "materialized": sum(1 for r in records if r.sync_status == "materialized"),
        }

    def _persist(self, record: TwinRecord) -> None:
        if self.memory is None:
            return
        with suppress(TypeError, ValueError, KeyError, AttributeError):
            self.memory.remember(
                "project",
                f"twin:{record.part_key}",
                record.summary(),
            )


__all__ = ["DigitalTwinSystem", "TwinRecord"]
