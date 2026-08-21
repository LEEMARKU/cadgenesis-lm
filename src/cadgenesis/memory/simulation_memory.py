"""cadgenesis.memory.simulation_memory
====================================
Simulation memory pool — past FEA/CFD results, safety factors and load cases,
enabling simulation-guided design refinement.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class SimulationMemory(MemoryStore):
    """Simulation results store (default 512 records)."""

    def __init__(
        self,
        capacity: int = 512,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name="simulation",
            capacity=capacity,
            default_importance=default_importance,
        )

    def remember_result(
        self,
        key: str,
        result: dict[str, Any],
        analysis_type: str = "structural",
        importance: float | None = None,
    ) -> MemoryEntry:
        """Store a simulation result with its analysis type."""
        return self.add(
            key,
            result,
            importance=importance,
            metadata={"kind": "result", "analysis_type": analysis_type},
        )

    def recall(
        self,
        query: str,
        top_k: int = 8,
        analysis_type: str | None = None,
    ) -> list[SearchResult]:
        hits = self.search(query, top_k=top_k)
        if analysis_type is None:
            return hits
        return [hit for hit in hits if hit.entry.metadata.get("analysis_type") == analysis_type]

    def by_analysis_type(self, analysis_type: str) -> list[MemoryEntry]:
        return [
            entry
            for entry in self.entries()
            if entry.metadata.get("analysis_type") == analysis_type
        ]
