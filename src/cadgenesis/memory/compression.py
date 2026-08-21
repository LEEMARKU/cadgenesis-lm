"""cadgenesis.memory.compression
===============================
Memory compression & consolidation (v6.0, Pillar 6).

Builds on the heuristic ``MemoryPruner`` with four complementary tools:

* :class:`MemorySummarizer` — merges related records into a single compact
  summary entry (lossy summarization).
* :class:`EmbeddingCompressor` — shrinks float payload lists via mean-pooling /
  stride decimation while preserving a cheap reconstruction error estimate.
* :class:`MemoryConsolidator` — folds working/session records into the
  long-term store (hierarchical consolidation).
* :class:`AdaptivePruner` — prunes per-pool using recency *and* importance
  with a configurable value threshold (learned-style adaptive eviction).

Everything is pure Python and works on any ``MemoryStore``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore, _tokens

_COMPRESSION_KIND = "compressed"


@dataclass
class CompressionReport:
    """Result of a compression / consolidation pass."""

    target: str
    source_pool: str
    consumed_keys: list[str]
    created_keys: list[str]

    @property
    def consumed(self) -> int:
        return len(self.consumed_keys)


class MemorySummarizer:
    """Merges related records into a compact summary entry."""

    def summarize(
        self,
        store: MemoryStore,
        keys: list[str],
        summary_key: str = "summary:group",
        group: str = "generic",
        importance: float = 1.0,
    ) -> MemoryEntry | None:
        """Compress ``keys`` into one record; originals stay untouched."""
        originals: list[MemoryEntry] = []
        for key in keys:
            entry = store.peek(key)
            if entry is not None:
                originals.append(entry)
        if not originals:
            return None
        compact = {
            "group": group,
            "count": len(originals),
            "keys": [e.key for e in originals],
            "contents": [e.content for e in originals],
            "importance": [e.importance for e in originals],
            "created_min": min(e.created_at for e in originals),
            "created_max": max(e.created_at for e in originals),
        }
        return store.add(
            summary_key,
            compact,
            importance=importance,
            metadata={"kind": _COMPRESSION_KIND, "group": group},
        )

    def expand(self, summary: MemoryEntry) -> list[Any]:
        """Re-inflate the compact record into its original contents."""
        if summary.metadata.get("kind") != _COMPRESSION_KIND:
            return [summary.content]
        payload = summary.content
        return list(payload.get("contents", []))


class EmbeddingCompressor:
    """Shrinks float payload lists (mean-pool / stride decimation)."""

    def compress(
        self,
        values: list[float],
        factor: int = 4,
        mode: str = "mean",
    ) -> list[float]:
        """Reduce ``values`` by ``factor`` using ``mean`` or ``stride`` pooling."""
        if factor <= 0:
            raise ValueError("factor must be > 0")
        if mode not in {"mean", "stride"}:
            raise ValueError(f"unknown mode {mode!r}; choose mean or stride")
        if len(values) <= 1 or factor == 1:
            return list(values)
        result: list[float] = []
        for start in range(0, len(values), factor):
            chunk = values[start : start + factor]
            if mode == "mean":
                result.append(sum(chunk) / len(chunk))
            else:
                result.append(chunk[0])
        return result

    def expansion_ratio(self, values: list[float], factor: int = 4) -> float:
        """Fraction of the original size retained after :meth:`compress`."""
        if not values:
            return 1.0
        return len(self.compress(values, factor=factor)) / len(values)

    def reconstruction_error(self, original: list[float], compressed: list[float]) -> float:
        """Mean absolute error of a nearest-index re-expansion."""
        if not original:
            return 0.0
        total = 0.0
        for index, value in enumerate(original):
            nearest = compressed[min(index, len(compressed) - 1)]
            total += abs(value - nearest)
        return total / len(original)


class MemoryConsolidator:
    """Folds short-term records into a long-term store (hierarchical memory)."""

    def consolidate(
        self,
        source: MemoryStore,
        target: MemoryStore,
        query: str | None = None,
        min_importance: float = 0.5,
        max_age: float | None = None,
        group: str = "consolidated",
        top_k: int = 64,
    ) -> CompressionReport:
        """Move valuable, relevant source records into ``target``.

        Only records whose importance is at least ``min_importance`` (and,
        when ``query`` is given, whose search score is positive) are folded.
        Each folded record is copied into the target with provenance metadata;
        the source records are removed.
        """
        now = time.time()
        consumed: list[str] = []
        created: list[str] = []
        if query:
            query_tokens = _tokens(query)
            selected = [
                hit.entry
                for hit in source.search(query, top_k=top_k)
                if _tokens(hit.entry.text()) & query_tokens
                and hit.entry.importance >= min_importance
                and (max_age is None or now - hit.entry.last_access <= max_age)
            ]
        else:
            selected = [
                entry
                for entry in source.entries()
                if entry.importance >= min_importance
                and (max_age is None or now - entry.last_access <= max_age)
            ]
        for entry in selected[:top_k]:
            metadata = dict(entry.metadata)
            metadata.update(
                {
                    "consolidated_from": source.name,
                    "group": group,
                }
            )
            target.add(
                entry.key,
                entry.content,
                importance=entry.importance,
                metadata=metadata,
            )
            source.remove(entry.key)
            consumed.append(entry.key)
            created.append(entry.key)
        return CompressionReport(
            target=target.name,
            source_pool=source.name,
            consumed_keys=consumed,
            created_keys=created,
        )


class AdaptivePruner:
    """Value-threshold pruner (recency + importance blend)."""

    def prune(
        self,
        store: MemoryStore,
        value_threshold: float = 0.2,
        recency_weight: float = 0.5,
        now: float | None = None,
    ) -> list[str]:
        """Evict records whose blended value falls below ``value_threshold``.

        value = recency_weight * normalized_recency
                + (1 - recency_weight) * importance
        """
        current = now if now is not None else time.time()
        ages = [current - entry.last_access for entry in store.values()]
        max_age = max(ages) if ages else 1.0
        evicted: list[str] = []
        for key in list(store):
            entry = store.peek(key)
            if entry is None:
                continue
            age = current - entry.last_access
            recency = 1.0 - (age / max_age) if max_age > 0 else 1.0
            value = recency_weight * recency + (1.0 - recency_weight) * entry.importance
            if value < value_threshold:
                store.remove(key)
                evicted.append(key)
        return evicted


__all__ = [
    "AdaptivePruner",
    "CompressionReport",
    "EmbeddingCompressor",
    "MemoryConsolidator",
    "MemorySummarizer",
]
