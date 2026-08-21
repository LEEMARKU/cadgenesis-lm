"""cadgenesis.memory.session_memory
==================================
Session memory pool — context scoped to the current design session.  Records
carry a ``session_id`` facet so multiple concurrent sessions never bleed into
each other.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class SessionMemory(MemoryStore):
    """Session-scoped context (default 128 records)."""

    def __init__(
        self,
        capacity: int = 128,
        default_importance: float = 0.9,
    ):
        super().__init__(
            name="session",
            capacity=capacity,
            default_importance=default_importance,
        )
        self.active_session: str | None = None

    def begin_session(self, session_id: str) -> None:
        """Switch the active session scope."""
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.active_session = session_id

    def end_session(self) -> None:
        """Clear the active session scope."""
        self.active_session = None

    def remember(
        self,
        key: str,
        content: Any,
        session_id: str | None = None,
    ) -> MemoryEntry:
        """Record a session-scoped entry tagged with the active session."""
        sid = session_id or self.active_session
        metadata: dict[str, Any] = {}
        if sid:
            metadata["session_id"] = sid
        return self.add(key, content, metadata=metadata)

    def recall(
        self,
        query: str,
        top_k: int = 8,
        session_id: str | None = None,
    ) -> list[SearchResult]:
        """Search records scoped to the active (or given) session."""
        sid = session_id or self.active_session
        hits = self.search(query, top_k=top_k)
        if sid is None:
            return hits
        return [hit for hit in hits if hit.entry.metadata.get("session_id") == sid]

    def session_entries(self, session_id: str | None = None) -> list[MemoryEntry]:
        """All records belonging to the active (or given) session."""
        sid = session_id or self.active_session
        if sid is None:
            return self.entries()
        return [entry for entry in self.entries() if entry.metadata.get("session_id") == sid]

    def clear_session(self, session_id: str | None = None) -> int:
        """Drop every record of the active (or given) session. Returns count."""
        sid = session_id or self.active_session
        if sid is None:
            return 0
        keys = [entry.key for entry in self.entries() if entry.metadata.get("session_id") == sid]
        for key in keys:
            self.remove(key)
        return len(keys)
