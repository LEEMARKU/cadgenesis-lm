"""cadgenesis.knowledge_network.sources
======================================
Concrete knowledge sources adapting the repository's symbolic artefacts.

* :class:`KnowledgeGraphSource` — wraps a
  :class:`~cadgenesis.reasoning.knowledge_graph.KnowledgeGraph`, scoring
  entries by node-label overlap with the query.
* :class:`StandardsSource` — wraps a
  :class:`~cadgenesis.reasoning.standards.StandardsLibrary`, scoring
  standards by identifier/title/scope overlap.
"""

from __future__ import annotations

from cadgenesis.knowledge_network.network import KnowledgeHit
from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph
from cadgenesis.reasoning.standards import StandardsLibrary


def _token_overlap(query: str, text: str) -> float:
    query_tokens = set(query.lower().split())
    text_tokens = set(text.lower().split())
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


class KnowledgeGraphSource:
    """A :class:`KnowledgeGraph` as a queryable knowledge source."""

    name = "knowledge_graph"

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def search(self, query: str, top_k: int = 8) -> list[KnowledgeHit]:
        hits: list[KnowledgeHit] = []
        for node in self.graph.nodes():
            label = f"{node.id} {node.label} {node.node_type}"
            score = _token_overlap(query, label)
            if score > 0.0:
                hits.append(
                    KnowledgeHit(
                        source=self.name,
                        identifier=node.id,
                        label=node.label or node.id,
                        score=score,
                        payload={
                            "node_type": node.node_type,
                            "attributes": dict(node.attributes),
                        },
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(top_k, 0)]

    def lookup(self, identifier: str) -> KnowledgeHit | None:
        node = self.graph.get_node(identifier)
        if node is None:
            return None
        return KnowledgeHit(
            source=self.name,
            identifier=node.id,
            label=node.label or node.id,
            score=1.0,
            payload={
                "node_type": node.node_type,
                "attributes": dict(node.attributes),
            },
        )

    def all(self) -> list[KnowledgeHit]:
        return [
            KnowledgeHit(
                source=self.name,
                identifier=node.id,
                label=node.label or node.id,
                score=1.0,
                payload={
                    "node_type": node.node_type,
                    "attributes": dict(node.attributes),
                },
            )
            for node in self.graph.nodes()
        ]


class StandardsSource:
    """A :class:`StandardsLibrary` as a queryable knowledge source."""

    name = "standards"

    def __init__(self, library: StandardsLibrary) -> None:
        self.library = library

    def search(self, query: str, top_k: int = 8) -> list[KnowledgeHit]:
        hits: list[KnowledgeHit] = []
        for identifier in self.library.identifiers:
            standard = self.library.get(identifier)
            if standard is None:
                continue
            text = f"{standard.identifier} {standard.title} {standard.scope} {standard.body}"
            score = _token_overlap(query, text)
            if score > 0.0:
                hits.append(
                    KnowledgeHit(
                        source=self.name,
                        identifier=standard.identifier,
                        label=standard.title or standard.identifier,
                        score=score,
                        payload={
                            "body": standard.body,
                            "kind": standard.kind,
                            "values": dict(standard.values),
                        },
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(top_k, 0)]

    def lookup(self, identifier: str) -> KnowledgeHit | None:
        standard = self.library.get(identifier)
        if standard is None:
            return None
        return KnowledgeHit(
            source=self.name,
            identifier=standard.identifier,
            label=standard.title or standard.identifier,
            score=1.0,
            payload={
                "body": standard.body,
                "kind": standard.kind,
                "values": dict(standard.values),
            },
        )

    def all(self) -> list[KnowledgeHit]:
        hits: list[KnowledgeHit] = []
        for identifier in self.library.identifiers:
            standard = self.library.get(identifier)
            if standard is None:
                continue
            hits.append(
                KnowledgeHit(
                    source=self.name,
                    identifier=standard.identifier,
                    label=standard.title or standard.identifier,
                    score=1.0,
                    payload={
                        "body": standard.body,
                        "kind": standard.kind,
                        "values": dict(standard.values),
                    },
                )
            )
        return hits


__all__ = [
    "KnowledgeGraphSource",
    "StandardsSource",
]
