"""tests/memory/test_pillar6_persistence.py
==========================================
Unit tests for the Pillar 6 persistence v2 features.
"""

from __future__ import annotations

import json

from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.persistence import MemoryPersistence


def test_v2_default_dumps():
    store = MemoryStore("t", capacity=8)
    store.add("a", 1)
    payload = json.loads(MemoryPersistence.dumps(store))
    assert payload["version"] == 2
    assert payload["schema"] == "v2"
    assert "written_at" in payload


def test_v1_payload_still_readable():
    store = MemoryStore("t", capacity=8)
    store.add("a", 1)
    payload = json.loads(MemoryPersistence.dumps(store, version=1))
    assert payload["version"] == 1
    restored = MemoryPersistence.loads(json.dumps(payload))
    assert restored.get("a").content == 1


def test_save_system_roundtrip(tmp_path):
    s1 = MemoryStore("a", capacity=8)
    s1.add("k1", "v1")
    s2 = MemoryStore("b", capacity=8)
    s2.add("k2", "v2")
    path = MemoryPersistence.save_system([s1, s2], tmp_path, label="daily")
    stores = MemoryPersistence.load_system(path)
    assert set(stores) == {"a", "b"}
    assert stores["a"].get("k1").content == "v1"
    assert stores["b"].get("k2").content == "v2"


def test_snapshot_rollback_restores_state():
    store = MemoryStore("t", capacity=8)
    store.add("k1", "old")
    snapshot = MemoryPersistence.snapshot([store])
    store.add("k2", "new")
    store.update("k1", content="mutated")
    restored = MemoryPersistence.rollback([store], snapshot)
    assert restored == ["t"]
    assert store.contains("k2") is False
    assert store.get("k1").content == "old"


def test_append_replay(tmp_path):
    store = MemoryStore("t", capacity=16)
    MemoryPersistence.append(store, "a", {"x": 1}, tmp_path)
    MemoryPersistence.append(store, "b", {"x": 2}, tmp_path)
    fresh = MemoryStore("t", capacity=16)
    replayed = MemoryPersistence.replay(tmp_path, fresh)
    assert replayed == ["a", "b"]
    assert fresh.get("a").content == {"x": 1}


def test_append_replay_since(tmp_path):
    store = MemoryStore("t", capacity=16)
    MemoryPersistence.append(store, "a", 1, tmp_path)
    timestamp = store.get("a").created_at
    MemoryPersistence.append(store, "b", 2, tmp_path)
    fresh = MemoryStore("t", capacity=16)
    replayed = MemoryPersistence.replay(tmp_path, fresh, since_timestamp=timestamp)
    assert replayed == ["b"]


def test_truncate_log(tmp_path):
    store = MemoryStore("t", capacity=16)
    MemoryPersistence.append(store, "a", 1, tmp_path)
    assert MemoryPersistence.truncate_log(tmp_path, store) is True
    assert MemoryPersistence.truncate_log(tmp_path, store) is False


def test_loads_handles_v2_system_document():
    s = MemoryStore("t", capacity=8)
    s.add("k", "v")
    payload = {
        "format": "cadgenesis-memory",
        "version": 2,
        "store": [s.to_dict()],
    }
    restored = MemoryPersistence.loads(json.dumps(payload))
    assert restored.name == "t"
    assert restored.get("k").content == "v"
