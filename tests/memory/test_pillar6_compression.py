"""tests/memory/test_pillar6_compression.py
==========================================
Unit tests for the Pillar 6 compression & consolidation tools.
"""

from __future__ import annotations

import pytest

from cadgenesis.memory.compression import (
    AdaptivePruner,
    EmbeddingCompressor,
    MemoryConsolidator,
    MemorySummarizer,
)
from cadgenesis.memory.long_term_memory import LongTermMemory
from cadgenesis.memory.memory_common import MemoryStore


def test_summarizer_merges_records():
    store = MemoryStore("working", capacity=64)
    store.add("a", "alpha record", importance=0.5)
    store.add("b", "beta record", importance=0.6)
    summarizer = MemorySummarizer()
    summary = summarizer.summarize(store, ["a", "b"], summary_key="sum:ab", group="g")
    assert summary is not None
    assert summary.content["count"] == 2
    assert summary.metadata["kind"] == "compressed"


def test_summarizer_empty_keys_returns_none():
    store = MemoryStore("working", capacity=8)
    summarizer = MemorySummarizer()
    assert summarizer.summarize(store, ["missing"]) is None


def test_summarizer_expand_roundtrip():
    store = MemoryStore("working", capacity=16)
    store.add("a", {"x": 1})
    store.add("b", {"x": 2})
    summarizer = MemorySummarizer()
    summary = summarizer.summarize(store, ["a", "b"])
    contents = summarizer.expand(summary)
    assert contents == [{"x": 1}, {"x": 2}]


def test_embedding_compressor_mean_pool():
    compressor = EmbeddingCompressor()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    compressed = compressor.compress(values, factor=2, mode="mean")
    assert compressed == [1.5, 3.5, 5.0]


def test_embedding_compressor_stride():
    compressor = EmbeddingCompressor()
    compressed = compressor.compress([1.0, 2.0, 3.0, 4.0], factor=2, mode="stride")
    assert compressed == [1.0, 3.0]


def test_embedding_compressor_rejects_bad_mode():
    compressor = EmbeddingCompressor()
    with pytest.raises(ValueError):
        compressor.compress([1.0], factor=2, mode="bogus")


def test_expansion_ratio_and_error():
    compressor = EmbeddingCompressor()
    original = [float(i) for i in range(32)]
    compressed = compressor.compress(original, factor=4)
    assert compressor.expansion_ratio(original, factor=4) == 0.25
    error = compressor.reconstruction_error(original, compressed)
    assert 0.0 <= error < 20.0


def test_consolidator_moves_valuable_records():
    source = MemoryStore("working", capacity=16)
    source.add("keep", "tolerance note", importance=0.9)
    source.add("drop", "noise", importance=0.1)
    target = LongTermMemory()
    consolidator = MemoryConsolidator()
    report = consolidator.consolidate(source, target, min_importance=0.5)
    assert report.consumed == 1
    assert report.created_keys == ["keep"]
    assert source.contains("drop")
    assert not source.contains("keep")
    assert target.contains("keep")


def test_consolidator_query_scoped():
    source = MemoryStore("working", capacity=16)
    source.add("m1", "machining tolerance for shafts", importance=0.8)
    source.add("u1", "user preference style", importance=0.9)
    target = LongTermMemory()
    consolidator = MemoryConsolidator()
    report = consolidator.consolidate(source, target, query="machining", min_importance=0.5)
    assert report.consumed == 1
    assert target.contains("m1")
    assert not target.contains("u1")


def test_adaptive_pruner_value_threshold():
    store = MemoryStore("working", capacity=16)
    store.add("keep", "important", importance=0.9)
    store.add("drop", "stale", importance=0.0)
    store.get("keep")
    pruner = AdaptivePruner()
    evicted = pruner.prune(store, value_threshold=0.3, now=store.peek("drop").last_access + 1000)
    assert "drop" in evicted
    assert "keep" not in evicted
