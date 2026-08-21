"""cadgenesis.cad.integration.simulation_bridge
=============================================
Bridge from the ``cad`` package to the simulation subsystem.

There is no standalone numerical simulation engine yet; simulation support in
CADGenesis-LM is represented by:

* :class:`SimulationMemory` (``memory/simulation_memory.py``) — stores past
  FEA/CFD results, safety factors and load cases;
* simulation token families (``SIM_*``) registered by the tokenizer.

This bridge maps CAD meshes / designs into simulation *requests* and *results*
so that the CAD layer can persist and retrieve analysis data consistently, and
can derive simple analytic proxies (e.g. mesh quality -> a coarse FEA
readiness score) without requiring an external solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, SearchResult
from cadgenesis.memory.simulation_memory import SimulationMemory

_ANALYSIS_TYPES = ("structural", "thermal", "fluid", "modal")


@dataclass
class SimulationSetup:
    """A request to run / record a simulation on a CAD object."""

    design_key: str
    analysis_type: str = "structural"
    load_case: str = "service"
    safety_factor: float = 1.5
    results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_key": self.design_key,
            "analysis_type": self.analysis_type,
            "load_case": self.load_case,
            "safety_factor": self.safety_factor,
            "results": dict(self.results),
        }


class SimulationBridge:
    """Record and retrieve simulation results for CAD designs."""

    def __init__(self, memory: SimulationMemory | None = None) -> None:
        self.memory = memory or SimulationMemory()

    def setup(
        self,
        design_key: str,
        analysis_type: str = "structural",
        load_case: str = "service",
        safety_factor: float = 1.5,
    ) -> SimulationSetup:
        if analysis_type not in _ANALYSIS_TYPES:
            raise ValueError(
                f"unsupported analysis type {analysis_type!r}; "
                f"expected one of {list(_ANALYSIS_TYPES)}"
            )
        return SimulationSetup(
            design_key=design_key,
            analysis_type=analysis_type,
            load_case=load_case,
            safety_factor=safety_factor,
        )

    def record(self, setup: SimulationSetup, results: dict[str, Any]) -> MemoryEntry:
        """Persist a simulation result for a design in simulation memory."""
        setup.results.update(results)
        return self.memory.remember_result(
            f"sim:{setup.analysis_type}:{setup.design_key}",
            setup.to_dict(),
            analysis_type=setup.analysis_type,
        )

    def recall(
        self, design_key: str, analysis_type: str | None = None, top_k: int = 8
    ) -> list[SearchResult]:
        """Recall simulation results, optionally filtered by analysis type."""
        return self.memory.recall(design_key, top_k=top_k, analysis_type=analysis_type)

    def mesh_readiness(self, mesh) -> dict[str, Any]:
        """Coarse FEA-readiness proxy for a triangle mesh (no solver required).

        Returns vertex/face counts, watertight status and an estimated cell
        budget; used as a lightweight stand-in until a real solver lands.
        """
        if mesh is None:
            return {"mesh_present": False, "readiness": 0.0}
        volume = mesh.volume()
        watertight = mesh.is_watertight()
        readiness = (
            1.0
            if (watertight and volume > 0 and mesh.face_count >= 4)
            else (0.5 if volume > 0 else 0.0)
        )
        return {
            "mesh_present": True,
            "vertices": mesh.vertex_count,
            "faces": mesh.face_count,
            "watertight": bool(watertight),
            "volume": float(volume),
            "readiness": readiness,
        }

    def to_report(self, design_key: str, analysis_type: str) -> dict[str, Any]:
        records = self.recall(design_key, analysis_type=analysis_type)
        return {
            "design_key": design_key,
            "analysis_type": analysis_type,
            "record_count": len(records),
            "records": [r.entry.content for r in records],
        }


__all__ = ["SimulationBridge", "SimulationSetup"]
