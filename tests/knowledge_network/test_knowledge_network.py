"""Tests for the Pillar 7 knowledge network package."""

from __future__ import annotations

import pytest

from cadgenesis.knowledge_network import (
    KnowledgeGraphSource,
    KnowledgeHit,
    KnowledgeNetwork,
    StandardsSource,
)
from cadgenesis.reasoning import KnowledgeGraph
from cadgenesis.reasoning.standards import default_standards_library


def _graph_source() -> KnowledgeGraphSource:
    graph = KnowledgeGraph()
    graph.add_node("material:steel", label="AISI 1045 steel", node_type="material")
    graph.add_node("process:machining", label="CNC machining", node_type="process")
    graph.add_edge("material:steel", "process:machining", "suitable_for")
    return KnowledgeGraphSource(graph)


def _standards_source() -> StandardsSource:
    return StandardsSource(default_standards_library())


def test_register_and_protocol() -> None:
    network = KnowledgeNetwork([_graph_source(), _standards_source()])
    assert network.source_count == 2
    assert "knowledge_graph" in network.source_names
    assert "standards" in network.source_names
    with pytest.raises(ValueError):
        network.register(_graph_source())


def test_merged_search() -> None:
    network = KnowledgeNetwork([_graph_source(), _standards_source()])
    hits = network.search("tolerance", top_k=5)
    assert hits
    for hit in hits:
        assert isinstance(hit, KnowledgeHit)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_search_source_filter() -> None:
    network = KnowledgeNetwork([_graph_source(), _standards_source()])
    hits = network.search("steel", sources=["knowledge_graph"])
    assert hits
    assert all(h.source == "knowledge_graph" for h in hits)


def test_lookup_across_sources() -> None:
    network = KnowledgeNetwork([_graph_source(), _standards_source()])
    hit = network.lookup("ISO 286-1")
    assert hit is not None
    assert hit.source == "standards"
    assert network.lookup("does-not-exist") is None


def test_all_and_stats() -> None:
    network = KnowledgeNetwork([_graph_source(), _standards_source()])
    all_hits = network.all()
    assert len(all_hits) >= 4
    stats = network.stats()
    assert stats["counts"]["knowledge_graph"] == 2


def test_to_graph_materialisation() -> None:
    network = KnowledgeNetwork([_graph_source()])
    graph = network.to_graph()
    assert graph.node_count >= 3  # 1 source node + 2 entry nodes
    assert graph.get_node("knowledge_graph") is not None
    assert graph.get_node("knowledge_graph:material:steel") is not None


def test_unregister() -> None:
    network = KnowledgeNetwork([_graph_source()])
    assert network.unregister("knowledge_graph") is True
    assert network.source_count == 0
    assert network.unregister("knowledge_graph") is False


def test_source_search_empty() -> None:
    source = _graph_source()
    assert source.search("zzzzy") == []
    assert source.lookup("nope") is None


def test_standards_source_search() -> None:
    source = _standards_source()
    hits = source.search("ISO")
    assert hits
    assert source.lookup("ASME B4.1") is not None
