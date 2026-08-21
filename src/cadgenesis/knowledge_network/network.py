"""cadgenesis.knowledge_network.network
======================================
The knowledge network core: a registry of typed knowledge sources with a
merged query surface.

A :class:`KnowledgeSource` is any object exposing ``name``, ``search(query,
top_k)``, ``lookup(identifier)`` and ``all()``.  :class:`KnowledgeNetwork`
registers sources, searches them in parallel (sequentially here; sources are
pure-Python), merges ranked hits, and can materialise the union of every
source into one :class:`~cadgenesis.reasoning.knowledge_graph.KnowledgeGraph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph


@dataclass
class KnowledgeHit:
    """A single ranked result from one knowledge source."""

    source: str
    identifier: str
    label: str = ""
    score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "label": self.label,
            "score": round(self.score, 4),
        }


class KnowledgeSource(Protocol):
    """Contract implemented by every knowledge source."""

    name: str

    def search(self, query: str, top_k: int = 8) -> list[KnowledgeHit]: ...

    def lookup(self, identifier: str) -> KnowledgeHit | None: ...

    def all(self) -> list[KnowledgeHit]: ...


class KnowledgeNetwork:
    """Merged, ranked query surface over many knowledge sources."""

    def __init__(self, sources: list[KnowledgeSource] | None = None) -> None:
        self._sources: dict[str, KnowledgeSource] = {}
        if sources:
            for source in sources:
                self.register(source)

    # ------------------------------------------------------------- registry

    def register(self, source: KnowledgeSource) -> None:
        if source.name in self._sources:
            raise ValueError(f"knowledge source {source.name!r} already registered")
        self._sources[source.name] = source

    def unregister(self, name: str) -> bool:
        return self._sources.pop(name, None) is not None

    def get(self, name: str) -> KnowledgeSource | None:
        return self._sources.get(name)

    @property
    def source_names(self) -> list[str]:
        return sorted(self._sources)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    # -------------------------------------------------------------- queries

    def search(
        self,
        query: str,
        top_k: int = 8,
        sources: list[str] | None = None,
    ) -> list[KnowledgeHit]:
        """Ranked, deduplicated search across (selected) sources."""
        selected = [s for name, s in self._sources.items() if sources is None or name in sources]
        merged: dict[str, KnowledgeHit] = {}
        for source in selected:
            for hit in source.search(query, top_k=top_k):
                existing = merged.get(hit.identifier)
                if existing is None or hit.score > existing.score:
                    merged[hit.identifier] = hit
        ranked = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return ranked[: max(top_k, 0)]

    def lookup(self, identifier: str) -> KnowledgeHit | None:
        """Look up an identifier across sources (best score wins)."""
        best: KnowledgeHit | None = None
        for source in self._sources.values():
            hit = source.lookup(identifier)
            if hit is not None and (best is None or hit.score > best.score):
                best = hit
        return best

    def all(self, sources: list[str] | None = None) -> list[KnowledgeHit]:
        hits: list[KnowledgeHit] = []
        for name, source in self._sources.items():
            if sources is not None and name not in sources:
                continue
            hits.extend(source.all())
        return hits

    # ----------------------------------------------------------- materialise

    def to_graph(self) -> KnowledgeGraph:
        """Union of all sources rendered into one KnowledgeGraph.

        Every source becomes a ``node_type="source"`` node linked to its
        entries (``node_type="entry"``) by a ``"provides"`` edge.
        """
        graph = KnowledgeGraph()
        for name, source in self._sources.items():
            graph.add_node(name, label=f"source {name}", node_type="source")
            for hit in source.all():
                node_id = f"{name}:{hit.identifier}"
                graph.add_node(
                    node_id,
                    label=hit.label or hit.identifier,
                    node_type="entry",
                    attributes={"identifier": hit.identifier, "score": hit.score},
                )
                graph.add_edge(name, node_id, "provides")
        return graph

    def stats(self) -> dict[str, Any]:
        return {
            "sources": self.source_names,
            "counts": {name: len(list(source.all())) for name, source in self._sources.items()},
        }


__all__ = [
    "KnowledgeHit",
    "KnowledgeNetwork",
    "KnowledgeSource",
]
