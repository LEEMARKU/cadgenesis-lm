"""tests/memory/test_pillar6_long_term.py
========================================
Unit tests for the Pillar 6 long-term memory store + system registration.
"""

from __future__ import annotations

import pytest

from cadgenesis.memory.long_term_memory import LONG_TERM_POOL, LongTermMemory
from cadgenesis.memory.memory_system import MemorySystem


def test_long_term_is_ninth_store():
    lt = LongTermMemory(capacity=128)
    assert lt.name == "long_term"
    assert lt.capacity == 128
    assert LONG_TERM_POOL == "long_term"


def test_consolidate_adds_provenance():
    lt = LongTermMemory()
    entry = lt.consolidate("p1", {"rule": "tolerance"}, source="iso2768")
    assert entry.metadata["kind"] == "consolidated"
    assert entry.metadata["source"] == "iso2768"
    assert lt.get("p1") is entry


def test_record_episode_and_episodes():
    lt = LongTermMemory()
    lt.record_episode("e1", "session summary", project_id="proj-a")
    lt.record_episode("e2", "another", importance=2.0)
    episodes = lt.episodes()
    assert len(episodes) == 2
    assert episodes[0].key == "e2"  # higher importance first


def test_recall_search():
    lt = LongTermMemory()
    lt.consolidate("c1", "lathe tolerance standard")
    hits = lt.recall("tolerance", top_k=5)
    assert hits
    assert hits[0].entry.key == "c1"


def test_default_system_keeps_eight_pools():
    system = MemorySystem()
    assert len(system.POOLS) == 8
    assert "long_term" not in system.POOLS
    assert "long_term" not in system.pools


def test_register_store_adds_ninth_pool():
    system = MemorySystem()
    lt = LongTermMemory()
    system.register_store(lt, keywords={"long", "term", "pattern"})
    assert len(system.POOLS) == 9
    assert "long_term" in system.pools
    assert system.pool("long_term") is lt
    system.remember("long_term", "k", {"v": 1})
    assert system.recall("long_term", "k").content == {"v": 1}


def test_registered_store_wired_into_retriever_and_router():
    system = MemorySystem()
    system.register_store(LongTermMemory())
    system.remember("long_term", "rule", "tolerance pattern for shafts", importance=0.9)
    result = system.retrieve("tolerance pattern", top_k=4)
    assert "long_term" in result.by_pool()
    routed = system.route("tolerance pattern")
    assert any(d.pool == "long_term" for d in routed)


def test_register_replaces_existing_store():
    system = MemorySystem()
    first = LongTermMemory()
    second = LongTermMemory()
    system.register_store(first)
    system.register_store(second)
    assert system.pool("long_term") is second


def test_register_rejects_unnamed_store():
    from cadgenesis.memory.memory_common import MemoryStore

    with pytest.raises(ValueError):
        MemoryStore("", capacity=4)
