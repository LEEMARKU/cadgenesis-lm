"""tests/memory/test_working_memory.py
=====================================
Unit tests for the working memory pool.
"""

from __future__ import annotations

from cadgenesis.memory.working_memory import WorkingMemory


def test_remember_recall():
    store = WorkingMemory(capacity=8)
    store.remember("w1", "active sketch")
    assert store.recall("sketch")[0].entry.key == "w1"


def test_context_returns_recent():
    store = WorkingMemory(capacity=8)
    store.remember("a", "first")
    store.remember("b", "second")
    assert store.context(top_k=2)[0].key in ("a", "b")


def test_squash_consumes():
    store = WorkingMemory(capacity=8)
    store.remember("w1", "data")
    assert store.squash("w1").key == "w1"
    assert "w1" not in store


def test_default_capacity():
    assert WorkingMemory().capacity == 64
