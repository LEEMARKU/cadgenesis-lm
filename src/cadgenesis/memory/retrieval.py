"""cadgenesis.memory.retrieval
=============================
Cross-pool retrieval over the semantic memory layer.

Combines ranked hits from multiple :class:`~cadgenesis.memory.memory_common.MemoryStore`
pools into a single, deduplicated, score-normalised result list — the
semantic analogue of the torch ``LayerIntegratedMemorySystem.retrieve``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, _tokens


@dataclass
class RetrievalHit:
    """A single cross-pool retrieval result."""

    entry: MemoryEntry
    pool: str
    score: float


@dataclass
class RetrievalResult:
    """Ordered cross-pool retrieval output."""

    query: str
    hits: list[RetrievalHit] = field(default_factory=list)

    @property
    def top(self) -> RetrievalHit | None:
        return self.hits[0] if self.hits else None

    def by_pool(self) -> dict[str, list[RetrievalHit]]:
        grouped: dict[str, list[RetrievalHit]] = {}
        for hit in self.hits:
            grouped.setdefault(hit.pool, []).append(hit)
        return grouped


class MemoryRetrieval:
    """Retrieval engine spanning any number of memory pools."""

    def __init__(self, stores: list[MemoryStore] | None = None):
        self._stores: dict[str, MemoryStore] = {}
        if stores:
            for store in stores:
                self.register(store)

    def register(self, store: MemoryStore) -> None:
        """Add (or replace) a pool to the retrieval index."""
        self._stores[store.name] = store

    def unregister(self, name: str) -> bool:
        return self._stores.pop(name, None) is not None

    @property
    def pool_names(self) -> list[str]:
        return sorted(self._stores)

    def stores(self) -> list[MemoryStore]:
        return list(self._stores.values())

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        pool_names: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> RetrievalResult:
        """Ranked, deduplicated retrieval across the requested pools.

        Each pool returns its best ``top_k`` matches; pool-level ``weights``
        scale those scores so a domain pool can be emphasised or muted.
        """
        names = pool_names or self.pool_names
        merged: dict[str, RetrievalHit] = {}
        for name in names:
            store = self._stores.get(name)
            if store is None:
                continue
            weight = (weights or {}).get(name, 1.0)
            for hit in store.search(query, top_k=top_k):
                existing = merged.get(hit.entry.key)
                score = hit.score * weight
                if existing is None or score > existing.score:
                    merged[hit.entry.key] = RetrievalHit(
                        entry=hit.entry,
                        pool=name,
                        score=score,
                    )
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return RetrievalResult(query=query, hits=hits[: max(top_k, 0)])

    def retrieve_multi(
        self,
        queries: list[str],
        top_k: int = 4,
        pool_names: list[str] | None = None,
    ) -> dict[str, RetrievalResult]:
        """Retrieve for several queries at once (dict keyed by query)."""
        return {
            query: self.retrieve(query, top_k=top_k, pool_names=pool_names) for query in queries
        }

    # ------------------------------------------------- P6 retrieval modes

    def graph_search(
        self,
        anchor_key: str,
        pool_names: list[str] | None = None,
        hop_count: int = 1,
        top_k: int = 8,
    ) -> RetrievalResult:
        """Follow ``metadata.related`` links from an anchor record.

        Records whose ``metadata`` carries a ``related`` list (keys in any
        registered pool) are traversed up to ``hop_count`` hops.  Results are
        ranked by path length (closer anchors rank higher).
        """
        names = pool_names or self.pool_names
        visited: set[str] = {anchor_key}
        frontier: list[tuple[str, int]] = [(anchor_key, 0)]
        collected: list[tuple[MemoryEntry, str, int]] = []
        while frontier:
            key, depth = frontier.pop(0)
            for name in names:
                store = self._stores.get(name)
                if store is None:
                    continue
                entry = store.peek(key)
                if entry is None:
                    continue
                if depth > 0:
                    collected.append((entry, name, depth))
                if depth < hop_count:
                    for related in entry.metadata.get("related") or []:
                        if related not in visited:
                            visited.add(related)
                            frontier.append((related, depth + 1))
        collected.sort(key=lambda item: (item[2], item[1]))
        hits = [
            RetrievalHit(entry=entry, pool=name, score=1.0 / (1.0 + depth))
            for entry, name, depth in collected
        ]
        return RetrievalResult(query=anchor_key, hits=hits[: max(top_k, 0)])

    def symbolic_search(
        self,
        constraints: dict[str, Any],
        pool_names: list[str] | None = None,
        top_k: int = 8,
    ) -> RetrievalResult:
        """Filter records by exact metadata facets (symbolic match).

        ``constraints`` maps metadata keys to expected values; a record
        matches when every key is present with an equal value.  An empty
        constraint dict matches the top most valuable records.
        """
        names = pool_names or self.pool_names
        hits: list[RetrievalHit] = []
        for name in names:
            store = self._stores.get(name)
            if store is None:
                continue
            for entry in store.values():
                meta = entry.metadata
                if all(meta.get(key) == value for key, value in constraints.items()):
                    hits.append(RetrievalHit(entry=entry, pool=name, score=entry.importance))
        if not constraints:
            hits.sort(key=lambda h: h.entry.importance, reverse=True)
        else:
            hits.sort(key=lambda h: h.score, reverse=True)
        return RetrievalResult(query=str(constraints), hits=hits[: max(top_k, 0)])

    def temporal_search(
        self,
        query: str,
        since: float | None = None,
        until: float | None = None,
        pool_names: list[str] | None = None,
        top_k: int = 8,
    ) -> RetrievalResult:
        """Keyword search restricted to a creation-time window.

        ``since``/``until`` are epoch seconds; records created inside the
        window are ranked by the same scorer used by the pool.
        """
        names = pool_names or self.pool_names
        query_tokens = _tokens(query)
        merged: dict[str, RetrievalHit] = {}
        for name in names:
            store = self._stores.get(name)
            if store is None:
                continue
            for hit in store.search(query, top_k=store.capacity):
                if not (_tokens(hit.entry.text()) & query_tokens):
                    continue
                created = hit.entry.created_at
                if since is not None and created < since:
                    continue
                if until is not None and created > until:
                    continue
                existing = merged.get(hit.entry.key)
                if existing is None or hit.score > existing.score:
                    merged[hit.entry.key] = RetrievalHit(
                        entry=hit.entry, pool=name, score=hit.score
                    )
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return RetrievalResult(query=query, hits=hits[: max(top_k, 0)])

    def hybrid_retrieve(
        self,
        query: str,
        symbolic: dict[str, Any] | None = None,
        temporal: tuple[float, float] | None = None,
        pool_names: list[str] | None = None,
        top_k: int = 8,
        keyword_weight: float = 0.7,
    ) -> RetrievalResult:
        """Combine keyword, symbolic-facet and temporal-window retrieval.

        Scores are the keyword score times ``keyword_weight`` plus the
        symbolic match bonus plus the temporal window bonus.
        """
        names = pool_names or self.pool_names
        merged: dict[str, RetrievalHit] = {}
        for name in names:
            store = self._stores.get(name)
            if store is None:
                continue
            hit_by_key = {
                hit.entry.key: hit.score for hit in store.search(query, top_k=store.capacity)
            }
            for entry in store.values():
                score = 0.0
                if entry.key in hit_by_key:
                    score += keyword_weight * hit_by_key[entry.key]
                if symbolic and all(
                    entry.metadata.get(key) == value for key, value in symbolic.items()
                ):
                    score += 1.0 - keyword_weight
                if temporal:
                    since, until = temporal
                    if since <= entry.created_at <= until:
                        score += (1.0 - keyword_weight) * 0.5
                if score > 0.0 or not symbolic:
                    existing = merged.get(entry.key)
                    if existing is None or score > existing.score:
                        merged[entry.key] = RetrievalHit(entry=entry, pool=name, score=score)
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return RetrievalResult(query=query, hits=hits[: max(top_k, 0)])

    def summary(self) -> dict[str, Any]:
        return {"pools": {name: store.summary() for name, store in self._stores.items()}}
