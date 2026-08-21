"""cadgenesis.continual_learning.replay_buffer
===========================================
Replay buffer for rehearsal-based continual learning.

Backed by the Pillar 6 semantic :class:`~cadgenesis.memory.memory_system.MemorySystem`:
experiences are stored as structured records in a designated pool and sampled
for rehearsal either uniformly or importance-weighted (so rare, high-value
experiences are replayed more often).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from cadgenesis.memory.memory_common import _tokens
from cadgenesis.memory.memory_system import MemorySystem

_REPLAY_POOL = "replay"


@dataclass
class ReplaySample:
    """A single replayed experience."""

    key: str
    content: Any
    importance: float
    metadata: dict[str, Any]


class ReplayBuffer:
    """Experience buffer persisted in a semantic memory pool."""

    def __init__(
        self,
        memory: MemorySystem,
        pool: str = _REPLAY_POOL,
        capacity: int = 2048,
        default_importance: float = 0.5,
    ) -> None:
        self.memory = memory
        self.pool = pool
        self.capacity = capacity
        self.default_importance = default_importance
        try:
            memory.pool(pool)
        except KeyError:
            from cadgenesis.memory.memory_common import MemoryStore

            memory.register_store(
                MemoryStore(pool, capacity=capacity),
                keywords={"replay", "experience", "episode"},
            )

    # -------------------------------------------------------------- writing

    def store(
        self,
        content: Any,
        importance: float | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record one experience; returns its key."""
        entry_key = key or f"exp:{len(self):06d}"
        self.memory.remember(
            self.pool,
            entry_key,
            content,
            importance=self.default_importance if importance is None else importance,
            metadata=metadata,
        )
        return entry_key

    def store_many(self, experiences: list[Any]) -> list[str]:
        return [self.store(exp) for exp in experiences]

    # -------------------------------------------------------------- sampling

    def sample(
        self,
        batch_size: int = 32,
        strategy: str = "importance",
    ) -> list[ReplaySample]:
        """Draw a rehearsal batch.

        ``strategy`` ∈ {uniform, importance}: importance sampling weights
        records by their stored importance.
        """
        pool = self.memory.pool(self.pool)
        entries = pool.entries()
        if not entries:
            return []
        size = min(batch_size, len(entries))
        if strategy == "uniform":
            chosen = random.sample(entries, size)
        elif strategy == "importance":
            weights = [e.importance for e in entries]
            chosen = random.choices(entries, weights=weights, k=size)
        else:
            raise ValueError(f"unknown sampling strategy {strategy!r}")
        return [
            ReplaySample(
                key=entry.key,
                content=entry.content,
                importance=entry.importance,
                metadata=dict(entry.metadata),
            )
            for entry in chosen
        ]

    def recall(self, query: str, top_k: int = 16) -> list[ReplaySample]:
        """Retrieve experiences relevant to a query (rehearsal by relevance)."""
        query_tokens = _tokens(query)
        hits = [
            hit
            for hit in self.memory.pool(self.pool).search(query, top_k=top_k)
            if _tokens(hit.entry.text()) & query_tokens
        ]
        return [
            ReplaySample(
                key=hit.entry.key,
                content=hit.entry.content,
                importance=hit.entry.importance,
                metadata=dict(hit.entry.metadata),
            )
            for hit in hits
        ]

    def clear(self) -> None:
        self.memory.pool(self.pool).clear()

    def __len__(self) -> int:
        return len(self.memory.pool(self.pool))


__all__ = ["ReplayBuffer", "ReplaySample"]
