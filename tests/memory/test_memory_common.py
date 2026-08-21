"""tests/memory/test_memory_common.py
=====================================
Unit tests for cadgenesis.memory.memory_common (base store + entry).
"""

from __future__ import annotations

import pytest

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore


def test_store_add_and_get():
    store = MemoryStore("test", capacity=10)
    store.add("a", {"part": "shaft"}, importance=0.9, metadata={"kind": "part"})
    entry = store.get("a")
    assert entry is not None
    assert entry.key == "a"
    assert entry.pool == "test"
    assert entry.importance == 0.9
    assert entry.metadata["kind"] == "part"


def test_store_overwrite_updates():
    store = MemoryStore("test", capacity=10)
    store.add("k", "v1")
    store.add("k", "v2")
    assert store.get("k").content == "v2"
    assert len(store) == 1


def test_store_capacity_enforced():
    store = MemoryStore("test", capacity=2)
    store.add("a", "x")
    store.add("b", "x")
    store.add("c", "x")
    assert len(store) <= 2


def test_store_remove_and_contains():
    store = MemoryStore("test", capacity=10)
    store.add("a", 1)
    assert store.remove("a") is True
    assert store.remove("a") is False
    assert "a" not in store


def test_store_update():
    store = MemoryStore("test", capacity=10)
    store.add("a", 1)
    assert store.update("a", content=2, importance=0.5) is True
    entry = store.peek("a")
    assert entry.content == 2
    assert entry.importance == 0.5
    assert store.update("missing", content=1) is False


def test_store_search_ranks_keyword_matches():
    store = MemoryStore("test", capacity=10)
    store.add("a", "extruded aluminum shaft")
    store.add("b", "injection molded bracket")
    hits = store.search("aluminum shaft")
    assert hits
    assert hits[0].entry.key == "a"
    assert hits[0].score > 0


def test_store_search_empty_query():
    store = MemoryStore("test", capacity=10)
    store.add("a", "anything")
    assert store.search("") == []


def test_store_top_returns_valuable():
    store = MemoryStore("test", capacity=10)
    store.add("low", "text", importance=0.1)
    store.add("high", "text", importance=0.9)
    ranked = store.top(top_k=1)
    assert ranked[0].key == "high"


def test_store_entry_touch_tracks_access():
    entry = MemoryEntry(key="k", content="c")
    before = entry.access_count
    entry.touch()
    assert entry.access_count == before + 1


def test_store_summary_shape():
    store = MemoryStore("test", capacity=5)
    store.add("a", "x")
    summary = store.summary()
    assert summary["name"] == "test"
    assert summary["size"] == 1
    assert summary["capacity"] == 5


def test_store_roundtrip_dict():
    store = MemoryStore("test", capacity=10)
    store.add("a", {"x": 1}, importance=0.7, metadata={"kind": "part"})
    restored = MemoryStore.from_dict(store.to_dict())
    entry = restored.get("a")
    assert entry.content == {"x": 1}
    assert entry.importance == 0.7
    assert entry.metadata == {"kind": "part"}


def test_store_rejects_bad_capacity():
    with pytest.raises(ValueError):
        MemoryStore("test", capacity=0)
    with pytest.raises(ValueError):
        MemoryStore("", capacity=5)
