"""tests/agents/test_shared_memory.py
===================================
Unit tests for cadgenesis.agents.shared_memory.
"""

from __future__ import annotations

from cadgenesis.agents.shared_memory import SharedMemory


def test_set_get():
    memory = SharedMemory()
    memory.set("k", "v")
    assert memory.get("k") == "v"
    assert memory.get("missing", "d") == "d"


def test_update_and_contains():
    memory = SharedMemory()
    memory.update({"a": 1, "b": 2})
    assert memory.contains("a")
    assert len(memory) == 2
    assert memory.items() == {"a": 1, "b": 2}


def test_remove():
    memory = SharedMemory()
    memory.set("k", "v")
    assert memory.remove("k") is True
    assert memory.remove("k") is False


def test_item_access():
    memory = SharedMemory()
    memory["k"] = 7
    assert memory["k"] == 7
    assert memory.get("k") == 7


def test_keys_snapshot():
    memory = SharedMemory()
    memory.set("a", 1)
    memory.set("b", 2)
    assert set(memory.keys) == {"a", "b"}


def test_clear():
    memory = SharedMemory()
    memory.set("a", 1)
    memory.clear()
    assert len(memory) == 0
