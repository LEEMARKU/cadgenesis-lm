"""cadgenesis.memory.pruning
===========================
Memory pruning / eviction policies for the semantic memory layer.

Policies:
    * ``capacity``   — trim the store to ``target_size`` keeping the most
      valuable records.
    * ``staleness``  — evict records whose ``last_access`` is older than
      ``max_age`` seconds.
    * ``importance`` — evict records below ``min_importance``.
    * ``combined``   — apply all of the above in one pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cadgenesis.memory.memory_common import MemoryStore


@dataclass
class PruningReport:
    """Result of a pruning pass."""

    store: str
    policy: str
    evicted: list[str]
    remaining: int

    @property
    def evicted_count(self) -> int:
        return len(self.evicted)


class MemoryPruner:
    """Stateless pruner applying eviction policies to memory pools."""

    # ----------------------------------------------------------- primitives

    @staticmethod
    def by_capacity(
        store: MemoryStore,
        target_size: int,
    ) -> list[str]:
        """Evict lowest-value records until ``target_size`` remains."""
        if target_size < 0:
            raise ValueError("target_size must be >= 0")
        if len(store) <= target_size:
            return []
        ranked = store.top(top_k=len(store))  # most valuable first
        keep = set(entry.key for entry in ranked[:target_size])
        evicted = [key for key in store if key not in keep]
        for key in evicted:
            store.remove(key)
        return evicted

    @staticmethod
    def by_staleness(
        store: MemoryStore,
        max_age: float,
        now: float | None = None,
    ) -> list[str]:
        """Evict records not accessed within ``max_age`` seconds."""
        current = now if now is not None else time.time()
        cutoff = current - max_age
        evicted = [
            key
            for key in store
            if store.peek(key).last_access < cutoff  # type: ignore[union-attr]
        ]
        for key in evicted:
            store.remove(key)
        return evicted

    @staticmethod
    def by_importance(
        store: MemoryStore,
        min_importance: float,
    ) -> list[str]:
        """Evict records with ``importance < min_importance``."""
        evicted = [
            key
            for key in store
            if store.peek(key).importance < min_importance  # type: ignore[union-attr]
        ]
        for key in evicted:
            store.remove(key)
        return evicted

    # ------------------------------------------------------------- dispatch

    def prune(
        self,
        store: MemoryStore,
        policy: str = "capacity",
        **params: Any,
    ) -> PruningReport:
        """Run a single named policy against a store."""
        if policy == "capacity":
            evicted = self.by_capacity(store, int(params.get("target_size", 0)))
        elif policy == "staleness":
            evicted = self.by_staleness(store, float(params.get("max_age", 60.0)))
        elif policy == "importance":
            evicted = self.by_importance(store, float(params.get("min_importance", 0.0)))
        elif policy == "combined":
            evicted = self._combined(store, params)
        else:
            raise ValueError(f"unknown pruning policy {policy!r}")
        return PruningReport(
            store=store.name,
            policy=policy,
            evicted=evicted,
            remaining=len(store),
        )

    def prune_all(
        self,
        stores: list[MemoryStore],
        policy: str = "capacity",
        **params: Any,
    ) -> list[PruningReport]:
        """Prune several stores; a keyword-scoped store list is also accepted."""
        return [self.prune(store, policy, **params) for store in stores]

    def _combined(self, store: MemoryStore, params: dict[str, Any]) -> list[str]:
        """Apply staleness, importance and capacity policies in one pass."""
        evicted: list[str] = []
        evicted.extend(self.by_staleness(store, float(params.get("max_age", 60.0))))
        evicted.extend(self.by_importance(store, float(params.get("min_importance", 0.0))))
        target = int(params.get("target_size", len(store)))
        evicted.extend(self.by_capacity(store, target))
        # Deduplicate (a key may be reported by several sub-policies).
        return list(dict.fromkeys(evicted))
