"""tests/memory/test_pillar6_retrieval.py
========================================
Unit tests for the Pillar 6 retrieval modes (graph/symbolic/temporal/hybrid).
"""

from __future__ import annotations

from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.retrieval import MemoryRetrieval


def build_retriever() -> MemoryRetrieval:
    store = MemoryStore("cad", capacity=64)
    store.add(
        "part:a",
        "block with tolerance",
        metadata={"kind": "part", "related": ["part:b", "mat:steel"]},
    )
    store.add(
        "part:b",
        "shaft",
        metadata={"kind": "part", "related": ["part:a"]},
    )
    store.add("mat:steel", "steel material", metadata={"kind": "material"})
    store.add("old:note", "stale record", metadata={"kind": "note"})
    retriever = MemoryRetrieval([store])
    return retriever


def test_graph_search_follows_links():
    retriever = build_retriever()
    result = retriever.graph_search("part:a", hop_count=1, top_k=8)
    keys = [h.entry.key for h in result.hits]
    assert "part:b" in keys
    assert "mat:steel" in keys
    assert "part:a" not in keys  # anchor itself is not reported


def test_graph_search_hop_depth_ranking():
    retriever = build_retriever()
    result = retriever.graph_search("part:a", hop_count=2, top_k=8)
    by_key = {h.entry.key: h.score for h in result.hits}
    assert by_key["part:b"] >= by_key["mat:steel"]


def test_symbolic_search_facet_match():
    retriever = build_retriever()
    result = retriever.symbolic_search({"kind": "part"})
    keys = {h.entry.key for h in result.hits}
    assert keys == {"part:a", "part:b"}


def test_symbolic_search_empty_constraints():
    retriever = build_retriever()
    result = retriever.symbolic_search({}, top_k=10)
    assert len(result.hits) == 4


def test_temporal_search_window():
    retriever = build_retriever()
    result = retriever.temporal_search("nonexistenttoken", since=0.0, until=10**13, top_k=8)
    assert len(result.hits) == 0  # no record contains the token


def test_temporal_search_keyword_in_window():
    retriever = build_retriever()
    result = retriever.temporal_search("tolerance", since=0.0, until=10**13, top_k=8)
    assert any(h.entry.key == "part:a" for h in result.hits)


def test_hybrid_retrieve_combines():
    retriever = build_retriever()
    result = retriever.hybrid_retrieve("tolerance", symbolic={"kind": "part"}, top_k=8)
    keys = {h.entry.key for h in result.hits}
    assert "part:a" in keys
    # symbolic-boosted part ranks above the pure-keyword match elsewhere
    assert result.hits[0].entry.key == "part:a"
