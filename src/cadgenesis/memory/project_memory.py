"""cadgenesis.memory.project_memory
==================================
Project memory pool — persistent design-state memory scoped to a project.
Holds versions, milestones, design intents and cross-file references so a
multi-file design session can be resumed.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class ProjectMemory(MemoryStore):
    """Project-scoped design state (default 512 records)."""

    def __init__(
        self,
        capacity: int = 512,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name="project",
            capacity=capacity,
            default_importance=default_importance,
        )
        self.project_id: str | None = None

    def attach(self, project_id: str) -> None:
        """Scope the store to a project."""
        if not project_id:
            raise ValueError("project_id must be non-empty")
        self.project_id = project_id

    def detach(self) -> None:
        self.project_id = None

    def remember(
        self,
        key: str,
        content: Any,
        project_id: str | None = None,
    ) -> MemoryEntry:
        pid = project_id or self.project_id
        metadata: dict[str, Any] = {}
        if pid:
            metadata["project_id"] = pid
        return self.add(key, content, metadata=metadata)

    def recall(
        self,
        query: str,
        top_k: int = 8,
        project_id: str | None = None,
    ) -> list[SearchResult]:
        pid = project_id or self.project_id
        hits = self.search(query, top_k=top_k)
        if pid is None:
            return hits
        return [hit for hit in hits if hit.entry.metadata.get("project_id") == pid]

    def snapshot(self, label: str, state: dict[str, Any]) -> MemoryEntry:
        """Persist a design-state snapshot under a label."""
        return self.add(
            f"snapshot:{label}",
            state,
            metadata={"kind": "snapshot"},
        )

    def last_snapshot(self) -> MemoryEntry | None:
        snapshots = [entry for entry in self.entries() if entry.metadata.get("kind") == "snapshot"]
        if not snapshots:
            return None
        return snapshots[-1]
