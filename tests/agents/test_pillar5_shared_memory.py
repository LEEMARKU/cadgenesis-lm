"""tests/agents/test_pillar5_shared_memory.py
=============================================
Unit tests for the Pillar 5 LayeredSharedMemory.
"""

from __future__ import annotations

import time

import pytest

from cadgenesis.agents.shared_memory import LayeredSharedMemory, SharedMemory


def test_shared_memory_legacy_api():
    memory = SharedMemory()
    memory.set("k", 1)
    memory["k2"] = 2
    assert memory.get("k") == 1
    assert memory["k2"] == 2
    assert memory.contains("k")
    assert len(memory) == 2
    memory.update({"k3": 3})
    assert len(memory) == 3
    assert memory.remove("k")
    assert memory.items() == {"k2": 2, "k3": 3}
    memory.clear()
    assert len(memory) == 0


def test_layered_region_defaults_and_set_get():
    memory = LayeredSharedMemory()
    assert "working" in memory.regions
    assert "agent" in memory.region_names()
    memory.set("working", "sketch", {"pts": [1, 2]})
    assert memory.get("working", "sketch") == {"pts": [1, 2]}
    assert memory.get("working", "missing", "d") == "d"


def test_layered_unknown_region():
    memory = LayeredSharedMemory()
    with pytest.raises(KeyError):
        memory.set("nope", "k", 1)


def test_layered_add_region():
    memory = LayeredSharedMemory()
    memory.add_region("debug", capacity=4, ttl=1.0)
    assert memory.exists_region("debug")
    with pytest.raises(ValueError):
        memory.add_region("debug")


def test_layered_capacity_evicts_oldest():
    memory = LayeredSharedMemory(default_capacity=2)
    memory.set("working", "a", 1)
    memory.set("working", "b", 2)
    memory.set("working", "c", 3)
    keys = sorted(memory.keys("working"))
    assert keys == ["b", "c"]
    assert memory.get("working", "a") is None


def test_layered_ttl_expiry():
    memory = LayeredSharedMemory(default_ttl=0.05)
    memory.set("working", "temp", "v")
    assert memory.get("working", "temp") == "v"
    time.sleep(0.1)
    assert memory.get("working", "temp") is None
    assert memory.keys("working") == []


def test_layered_change_notification():
    memory = LayeredSharedMemory()
    events = []
    memory.on_change("working", lambda region, key, value: events.append((region, key, value)))
    memory.set("working", "k", 1)
    assert events == [("working", "k", 1)]


def test_layered_snapshot_and_contains():
    memory = LayeredSharedMemory()
    memory.set("working", "k", 1)
    memory.set("project", "p", 2)
    assert "k" in memory
    snapshot = memory.snapshot()
    assert snapshot["working"]["k"] == 1
    assert memory.snapshot("project")["project"]["p"] == 2
    assert memory.usage()["working"] == 1


def test_layered_cache():
    memory = LayeredSharedMemory()
    memory.cache_put("token", 123, ttl=1.0)
    assert memory.cache_get("token") == 123
    memory.cache_clear()
    assert memory.cache_get("token") is None


def test_layered_mirror_into_memory_system():
    from cadgenesis.memory import MemorySystem

    system = MemorySystem()
    memory = LayeredSharedMemory(memory_system=system, memory_pool="project")
    memory.set("working", "k", {"v": 1})
    entry = system.recall("project", "working:k")
    assert entry.content == {"v": 1}


def test_layered_attach_memory():
    from cadgenesis.memory import MemorySystem

    memory = LayeredSharedMemory()
    memory.attach_memory(MemorySystem(), pool="project")
    memory.set("session", "note", "hello")
    assert memory.get("session", "note") == "hello"
