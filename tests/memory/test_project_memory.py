"""tests/memory/test_project_memory.py
=====================================
Unit tests for the project memory pool.
"""

from __future__ import annotations

import pytest

from cadgenesis.memory.project_memory import ProjectMemory


def test_attach_remember_recall():
    store = ProjectMemory(capacity=16)
    store.attach("proj-1")
    store.remember("v1", {"revision": 1})
    assert store.recall("revision")[0].entry.key == "v1"


def test_snapshots():
    store = ProjectMemory(capacity=16)
    store.attach("proj-1")
    store.snapshot("alpha", {"state": "ok"})
    store.snapshot("beta", {"state": "better"})
    assert store.last_snapshot().metadata["kind"] == "snapshot"
    assert store.last_snapshot().key == "snapshot:beta"


def test_detach():
    store = ProjectMemory(capacity=16)
    store.attach("proj-1")
    store.remember("v1", {"revision": 1})
    store.detach()
    assert store.project_id is None
    assert store.recall("revision")[0].entry.key == "v1"


def test_requires_project_id():
    store = ProjectMemory(capacity=16)
    with pytest.raises(ValueError):
        store.attach("")
