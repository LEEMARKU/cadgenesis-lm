"""cadgenesis.memory.memory_common
=================================
Shared foundation for the CADGenesis-LM semantic memory layer.

Defines the :class:`MemoryEntry` record and the :class:`MemoryStore` base class
that every domain memory pool (working, session, user, project, cad,
engineering, manufacturing, simulation) specialises.  The store is a
pure-Python, dependency-free keyed store with bounded capacity, importance,
recency tracking and lightweight keyword scoring — no torch required, so the
memory layer can be unit-tested and used standalone.

Unlike the torch-based :class:`~cadgenesis.memory.memory_pools.MemoryPool`,
which holds differentiable slot vectors for layer-integrated attention, this
layer stores *structured records* (CAD feature trees, standards, DFM limits,
simulation results, user preferences, ...).  The two are complementary:
``memory_pools`` is the neural bank, this module is the semantic bank.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class MemoryEntry:
    """A single structured memory record inside a :class:`MemoryStore`.

    ``content`` holds the payload (any JSON-serialisable object — a feature
    tree dict, a standard text, a simulation result, ...).  ``metadata``
    carries structured facets used for filtering and scoring.
    """

    key: str
    content: Any
    pool: str = ""
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self) -> None:
        """Record an access (recency + count)."""
        self.last_access = time.time()
        self.access_count += 1

    def text(self) -> str:
        """Flatten content + metadata into a searchable text blob."""
        parts = [str(self.content)]
        parts.extend(str(v) for v in self.metadata.values())
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation of the record."""
        return {
            "key": self.key,
            "content": self.content,
            "pool": self.pool,
            "importance": self.importance,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
        }


@dataclass
class SearchResult:
    """A scored hit returned by :meth:`MemoryStore.search`."""

    entry: MemoryEntry
    score: float


class MemoryStore:
    """Bounded, scored, keyed memory store (semantic bank base class).

    Subclasses (e.g. ``WorkingMemory``) set a default ``name`` / ``capacity``
    and may add domain helpers; all storage, scoring and eviction machinery
    lives here.
    """

    def __init__(
        self,
        name: str,
        capacity: int = 256,
        default_importance: float = 1.0,
        keyword_weight: float = 0.5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if not name:
            raise ValueError("memory store requires a name")
        self.name = name
        self.capacity = capacity
        self.default_importance = default_importance
        self.keyword_weight = keyword_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self._entries: dict[str, MemoryEntry] = {}
        self._created = time.time()

    # ------------------------------------------------------------------ write

    def add(
        self,
        key: str,
        content: Any,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Insert or overwrite a record.  Returns the stored entry."""
        entry = MemoryEntry(
            key=key,
            content=content,
            pool=self.name,
            importance=self.default_importance if importance is None else importance,
            metadata=dict(metadata or {}),
        )
        self._entries[key] = entry
        self._enforce_capacity()
        return entry

    def update(
        self,
        key: str,
        content: Any | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Patch an existing record; returns False when the key is absent."""
        entry = self._entries.get(key)
        if entry is None:
            return False
        if content is not None:
            entry.content = content
        if importance is not None:
            entry.importance = importance
        if metadata is not None:
            entry.metadata.update(metadata)
        return True

    def remove(self, key: str) -> bool:
        """Delete a record; returns True when it existed."""
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every record."""
        self._entries.clear()

    # ------------------------------------------------------------------- read

    def get(self, key: str) -> MemoryEntry | None:
        """Fetch a record and touch its recency."""
        entry = self._entries.get(key)
        if entry is not None:
            entry.touch()
        return entry

    def peek(self, key: str) -> MemoryEntry | None:
        """Fetch a record without updating recency."""
        return self._entries.get(key)

    def contains(self, key: str) -> bool:
        return key in self._entries

    def keys(self) -> Iterable[str]:
        return self._entries.keys()

    def values(self) -> Iterable[MemoryEntry]:
        return self._entries.values()

    def entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(self._entries)

    @property
    def is_full(self) -> bool:
        return len(self._entries) >= self.capacity

    # ---------------------------------------------------------------- scoring

    def _score(self, entry: MemoryEntry, query_tokens: set[str]) -> float:
        text_tokens = _tokens(entry.text())
        overlap = len(query_tokens & text_tokens)
        keyword = overlap / max(len(query_tokens), 1)
        age = time.time() - entry.last_access
        recency = 1.0 / (1.0 + age)
        return (
            self.keyword_weight * keyword
            + self.recency_weight * recency
            + self.importance_weight * entry.importance
        )

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Rank records by keyword overlap + recency + importance."""
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored = [
            SearchResult(entry=entry, score=self._score(entry, query_tokens))
            for entry in self._entries.values()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: max(top_k, 0)]

    def top(self, top_k: int = 10) -> list[MemoryEntry]:
        """Return the most valuable records (importance-weighted recency)."""
        ranked = sorted(
            self._entries.values(),
            key=lambda e: self._score(e, _tokens(e.text())),
            reverse=True,
        )
        return ranked[: max(top_k, 0)]

    # ------------------------------------------------------------------ misc

    def _enforce_capacity(self) -> None:
        """Evict the least valuable record when the store is over capacity."""
        while len(self._entries) > self.capacity:
            victim = min(
                self._entries.values(),
                key=lambda e: self._score(e, _tokens(e.text())),
            )
            self._entries.pop(victim.key, None)

    def summary(self) -> dict[str, Any]:
        """Snapshot of store state for telemetry / reports."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "size": len(self._entries),
            "created_at": self._created,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (see ``memory.persistence``)."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "default_importance": self.default_importance,
            "keyword_weight": self.keyword_weight,
            "recency_weight": self.recency_weight,
            "importance_weight": self.importance_weight,
            "entries": [
                {
                    "key": entry.key,
                    "content": entry.content,
                    "pool": entry.pool,
                    "importance": entry.importance,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at,
                    "last_access": entry.last_access,
                    "access_count": entry.access_count,
                }
                for entry in self._entries.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryStore:
        """Rebuild a store from :meth:`to_dict` output."""
        store = cls(
            name=str(data.get("name", "")),
            capacity=int(data.get("capacity", 256)),
            default_importance=float(data.get("default_importance", 1.0)),
            keyword_weight=float(data.get("keyword_weight", 0.5)),
            recency_weight=float(data.get("recency_weight", 0.3)),
            importance_weight=float(data.get("importance_weight", 0.2)),
        )
        for raw in data.get("entries", []):
            entry = MemoryEntry(
                key=str(raw["key"]),
                content=raw["content"],
                pool=str(raw.get("pool", store.name)),
                importance=float(raw.get("importance", 1.0)),
                metadata=dict(raw.get("metadata", {})),
                created_at=float(raw.get("created_at", time.time())),
                last_access=float(raw.get("last_access", time.time())),
                access_count=int(raw.get("access_count", 0)),
            )
            store._entries[entry.key] = entry
        return store
