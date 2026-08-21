"""cadgenesis.memory.long_term_memory
====================================
Long-term memory pool — the ninth semantic store (v6.0, Pillar 6).

While the working / session / project stores capture ephemeral and active
state, the long-term store holds consolidated, high-importance knowledge that
survives sessions and projects: learned design patterns, distilled
conventions, cross-project best practices.  It is registered *additively*
(see :meth:`~cadgenesis.memory.memory_system.MemorySystem.register_store`) so
the default 8-pool contract of the facade is untouched until a caller opts in.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult

LONG_TERM_POOL = "long_term"
"""Canonical name of the long-term semantic store."""

_EPISODE_MARKER = "episode"


class LongTermMemory(MemoryStore):
    """High-capacity, high-importance consolidated knowledge store."""

    def __init__(
        self,
        capacity: int = 4096,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name=LONG_TERM_POOL,
            capacity=capacity,
            default_importance=default_importance,
        )

    # ------------------------------------------------------------ write

    def consolidate(
        self,
        key: str,
        content: Any,
        source: str = "",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a consolidated knowledge record with provenance metadata."""
        meta = dict(metadata or {})
        meta["kind"] = "consolidated"
        if source:
            meta["source"] = source
        return self.add(
            key,
            content,
            importance=importance,
            metadata=meta,
        )

    def record_episode(
        self,
        key: str,
        summary: Any,
        project_id: str | None = None,
        importance: float | None = None,
    ) -> MemoryEntry:
        """Persist a distilled episode summary (session/event footprint)."""
        metadata: dict[str, Any] = {"kind": _EPISODE_MARKER}
        if project_id:
            metadata["project_id"] = project_id
        return self.add(key, summary, importance=importance, metadata=metadata)

    # ------------------------------------------------------------ read

    def recall(self, query: str, top_k: int = 8) -> list[SearchResult]:
        """Search consolidated knowledge records."""
        return self.search(query, top_k=top_k)

    def episodes(self, top_k: int = 64) -> list[MemoryEntry]:
        """All recorded episode summaries, most valuable first."""
        ranked = [
            entry for entry in self.entries() if entry.metadata.get("kind") == _EPISODE_MARKER
        ]
        ranked.sort(key=lambda e: (e.importance, e.last_access), reverse=True)
        return ranked[: max(top_k, 0)]


__all__ = ["LONG_TERM_POOL", "LongTermMemory"]
