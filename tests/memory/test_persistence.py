"""tests/memory/test_persistence.py
==================================
Unit tests for memory pool persistence / load.
"""

from __future__ import annotations

import json

import pytest

from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.persistence import MemoryPersistence

_TEST_DIR = "outputs/tests/memory"


def test_dumps_loads_roundtrip(tmp_path):
    store = MemoryStore("t", capacity=8)
    store.add("a", {"x": 1}, importance=0.6, metadata={"kind": "part"})
    text = MemoryPersistence.dumps(store)
    payload = json.loads(text)
    assert payload["format"] == "cadgenesis-memory"
    restored = MemoryPersistence.loads(text)
    assert restored.get("a").content == {"x": 1}
    assert restored.get("a").importance == 0.6


def test_save_load_file(tmp_path):
    store = MemoryStore("t", capacity=8)
    store.add("a", "hello")
    path = tmp_path / "pool.json"
    MemoryPersistence.save(store, path)
    assert path.exists()
    restored = MemoryPersistence.load(path)
    assert restored.get("a").content == "hello"


def test_save_creates_parent_dirs(tmp_path):
    store = MemoryStore("t", capacity=8)
    path = tmp_path / "deep" / "nested" / "pool.json"
    MemoryPersistence.save(store, path)
    assert path.exists()


def test_save_many_load_many(tmp_path):
    s1 = MemoryStore("a", capacity=8)
    s1.add("k", "v")
    s2 = MemoryStore("b", capacity=8)
    s2.add("k2", "v2")
    paths = MemoryPersistence.save_many([s1, s2], tmp_path)
    assert len(paths) == 2
    stores = MemoryPersistence.load_many(tmp_path)
    assert set(stores) == {"a", "b"}
    assert stores["a"].get("k").content == "v"


def test_load_rejects_wrong_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format": "other", "version": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        MemoryPersistence.load(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        MemoryPersistence.load(tmp_path / "nope.json")


def test_load_many_by_name(tmp_path):
    s1 = MemoryStore("a", capacity=8)
    s1.add("k", "v")
    MemoryPersistence.save_many([s1], tmp_path)
    stores = MemoryPersistence.load_many(tmp_path, names=["a"])
    assert "a" in stores
    stores = MemoryPersistence.load_many(tmp_path, names=["zzz"])
    assert stores == {}
