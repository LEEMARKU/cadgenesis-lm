"""tests/memory/test_session_memory.py
=====================================
Unit tests for the session memory pool.
"""

from __future__ import annotations

import pytest

from cadgenesis.memory.session_memory import SessionMemory


def test_session_scoping():
    store = SessionMemory(capacity=16)
    store.begin_session("s1")
    store.remember("a", "toolbar state", session_id="s1")
    store.begin_session("s2")
    store.remember("b", "panel state", session_id="s2")
    assert store.session_entries("s1")[0].key == "a"
    assert len(store.session_entries("s2")) == 1


def test_recall_scoped():
    store = SessionMemory(capacity=16)
    store.begin_session("s1")
    store.remember("a", "flange sketch", session_id="s1")
    store.begin_session("s2")
    store.remember("b", "flange sketch", session_id="s2")
    hits = store.recall("flange", session_id="s1")
    assert [h.entry.key for h in hits] == ["a"]


def test_clear_session():
    store = SessionMemory(capacity=16)
    store.begin_session("s1")
    store.remember("a", "x", session_id="s1")
    assert store.clear_session("s1") == 1
    assert "a" not in store


def test_requires_session_id():
    store = SessionMemory(capacity=16)
    with pytest.raises(ValueError):
        store.begin_session("")
