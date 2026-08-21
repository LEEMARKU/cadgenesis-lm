"""tests/memory/test_pruning.py
==============================
Unit tests for memory pruning / eviction policies.
"""

from __future__ import annotations

import time

import pytest

from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.pruning import MemoryPruner, PruningReport


def test_by_capacity_keeps_valuable():
    store = MemoryStore("test", capacity=10)
    store.add("low", "x", importance=0.1)
    store.add("high", "x", importance=0.9)
    evicted = MemoryPruner.by_capacity(store, target_size=1)
    assert "high" not in evicted
    assert len(store) == 1
    assert store.contains("high")


def test_by_staleness():
    store = MemoryStore("test", capacity=10)
    store.add("old", "x")
    store.add("fresh", "x")
    store.get("fresh")  # bump recency
    old = store.peek("old")
    fresh = store.peek("fresh")
    assert old is not None and fresh is not None
    old.last_access = time.time() - 3600.0  # pretend it aged out
    evicted = MemoryPruner.by_staleness(store, max_age=60.0)
    assert "old" in evicted
    assert "fresh" not in evicted


def test_by_importance():
    store = MemoryStore("test", capacity=10)
    store.add("low", "x", importance=0.2)
    store.add("keep", "x", importance=0.8)
    evicted = MemoryPruner.by_importance(store, min_importance=0.5)
    assert evicted == ["low"]
    assert "keep" in store


def test_prune_dispatch():
    store = MemoryStore("test", capacity=10)
    store.add("a", "x", importance=0.1)
    report = MemoryPruner().prune(store, policy="importance", min_importance=0.5)
    assert isinstance(report, PruningReport)
    assert report.evicted == ["a"]
    assert report.remaining == 0


def test_prune_unknown_policy_raises():
    with pytest.raises(ValueError):
        MemoryPruner().prune(MemoryStore("t"), policy="bogus")


def test_prune_combined_dedupes():
    store = MemoryStore("test", capacity=10)
    store.add("a", "x", importance=0.1)
    report = MemoryPruner().prune(store, policy="combined", min_importance=0.5, target_size=0)
    assert report.evicted == ["a"]


def test_prune_all():
    s1 = MemoryStore("s1", capacity=10)
    s1.add("x", "x", importance=0.1)
    s2 = MemoryStore("s2", capacity=10)
    s2.add("y", "y", importance=0.1)
    reports = MemoryPruner().prune_all([s1, s2], policy="importance", min_importance=0.5)
    assert [r.store for r in reports] == ["s1", "s2"]
