"""cadgenesis.memory.engineering_memory
=====================================
Engineering memory pool — ISO / ASME / DIN standards, design guidelines and
material data used to keep generated designs standards-compliant.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class EngineeringMemory(MemoryStore):
    """Engineering knowledge store (default 512 records)."""

    def __init__(
        self,
        capacity: int = 512,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name="engineering",
            capacity=capacity,
            default_importance=default_importance,
        )

    def remember_standard(
        self,
        standard_id: str,
        body: Any,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a standard (ISO/ASME/DIN, guideline, or material table)."""
        meta = dict(metadata or {})
        meta.setdefault("kind", "standard")
        return self.add(f"standard:{standard_id}", body, metadata=meta)

    def standard(self, standard_id: str) -> Any | None:
        entry = self.peek(f"standard:{standard_id}")
        if entry is None:
            return None
        entry.touch()
        return entry.content

    def recall(self, query: str, top_k: int = 8) -> list[SearchResult]:
        return self.search(query, top_k=top_k)

    def guidelines(self) -> list[MemoryEntry]:
        return [entry for entry in self.entries() if entry.metadata.get("kind") == "guideline"]
