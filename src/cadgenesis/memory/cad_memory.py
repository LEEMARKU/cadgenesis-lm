"""cadgenesis.memory.cad_memory
==============================
CAD memory pool — historical feature trees, B-Rep topologies and parametric
design data.  The largest pool; drives design reuse and retrieval-augmented
generation of geometry.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, SearchResult


class CADMemory(MemoryStore):
    """CAD object / design memory (default 1024 records)."""

    def __init__(
        self,
        capacity: int = 1024,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name="cad",
            capacity=capacity,
            default_importance=default_importance,
        )

    def remember_feature_tree(
        self,
        key: str,
        feature_tree: list[dict[str, Any]],
        kind: str = "part",
        importance: float | None = None,
    ) -> MemoryEntry:
        """Store a parametric feature tree."""
        return self.add(
            key,
            feature_tree,
            importance=importance,
            metadata={"kind": "feature_tree", "object_kind": kind},
        )

    def remember_brep(
        self,
        key: str,
        brep: dict[str, Any],
        kind: str = "part",
        importance: float | None = None,
    ) -> MemoryEntry:
        """Store a boundary-representation topology."""
        return self.add(
            key,
            brep,
            importance=importance,
            metadata={"kind": "brep", "object_kind": kind},
        )

    def recall(
        self,
        query: str,
        top_k: int = 8,
        object_kind: str | None = None,
    ) -> list[SearchResult]:
        hits = self.search(query, top_k=top_k)
        if object_kind is None:
            return hits
        return [hit for hit in hits if hit.entry.metadata.get("object_kind") == object_kind]

    def by_kind(self, kind: str) -> list[MemoryEntry]:
        return [entry for entry in self.entries() if entry.metadata.get("object_kind") == kind]
