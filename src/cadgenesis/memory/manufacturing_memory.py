"""cadgenesis.memory.manufacturing_memory
=======================================
Manufacturing memory pool — machining and process limits (DFM), machine
capabilities and shop-floor constraints used to validate manufacturability.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class ManufacturingMemory(MemoryStore):
    """Manufacturing knowledge store (default 512 records)."""

    def __init__(
        self,
        capacity: int = 512,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name="manufacturing",
            capacity=capacity,
            default_importance=default_importance,
        )

    def remember_process(
        self,
        process: str,
        limits: dict[str, Any],
    ) -> MemoryEntry:
        """Store a process capability / limit record."""
        return self.add(
            f"process:{process}",
            limits,
            metadata={"kind": "process", "process": process},
        )

    def process_limits(self, process: str) -> dict[str, Any] | None:
        entry = self.peek(f"process:{process}")
        if entry is None:
            return None
        entry.touch()
        return entry.content

    def recall(self, query: str, top_k: int = 8) -> list[SearchResult]:
        return self.search(query, top_k=top_k)

    def processes(self) -> list[str]:
        return [
            entry.metadata["process"]
            for entry in self.entries()
            if entry.metadata.get("kind") == "process"
        ]
