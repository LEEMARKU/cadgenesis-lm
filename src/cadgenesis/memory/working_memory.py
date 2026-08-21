"""cadgenesis.memory.working_memory
=================================
Working memory pool — the short-term context buffer for the active design
generation.  Small capacity, recency-weighted, and the only pool that the
generation loop rewrites continuously.
"""

from __future__ import annotations

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class WorkingMemory(MemoryStore):
    """Ephemeral context buffer (default 64 records)."""

    def __init__(
        self,
        capacity: int = 64,
        default_importance: float = 0.8,
    ):
        super().__init__(
            name="working",
            capacity=capacity,
            default_importance=default_importance,
        )

    def remember(self, key: str, content: object) -> MemoryEntry:
        """Add a working-context record."""
        return self.add(key, content)

    def recall(self, query: str, top_k: int = 8) -> list[SearchResult]:
        """Retrieve the most relevant working records."""
        return self.search(query, top_k=top_k)

    def context(self, top_k: int = 8) -> list[MemoryEntry]:
        """The current short-term context (most recently accessed first)."""
        return self.top(top_k=top_k)

    def squash(self, key: str) -> MemoryEntry | None:
        """Retrieve and remove a record in one step (consume pattern)."""
        entry = self.get(key)
        if entry is not None:
            self.remove(key)
        return entry
