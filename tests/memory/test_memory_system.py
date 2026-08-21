"""tests/memory/test_memory_system.py
====================================
Unit tests for the MemorySystem facade.
"""

from __future__ import annotations

import pytest

from cadgenesis.config import MemoryConfig
from cadgenesis.memory.memory_system import MemorySystem


def test_system_has_eight_pools():
    system = MemorySystem()
    assert set(system.pools) == {
        "working",
        "session",
        "user",
        "project",
        "cad",
        "engineering",
        "manufacturing",
        "simulation",
    }


def test_system_pool_access():
    system = MemorySystem()
    assert system.pool("cad") is system.cad
    with pytest.raises(KeyError):
        system.pool("bogus")


def test_system_remember_retrieve():
    system = MemorySystem()
    system.remember("cad", "part:x", {"op": "extrude"})
    assert system.recall("cad", "part:x").content == {"op": "extrude"}
    result = system.retrieve("extrude")
    assert result.top is not None
    assert result.top.entry.key == "part:x"


def test_system_forget():
    system = MemorySystem()
    system.remember("cad", "part:x", "data")
    assert system.forget("cad", "part:x") is True
    assert system.forget("cad", "part:x") is False


def test_system_route():
    system = MemorySystem()
    system.cad.remember_feature_tree("part:flange", [{"op": "extrude"}])
    assert system.route("extrude feature")[0].pool == "cad"


def test_system_prune():
    system = MemorySystem()
    system.cad.add("low", "x", importance=0.1)
    reports = system.prune(pool_names=["cad"], policy="importance", min_importance=0.5)
    assert reports[0].evicted == ["low"]


def test_system_from_config():
    cfg = MemoryConfig(working_memory_slots=8, cad_memory_slots=8)
    system = MemorySystem.from_config(cfg)
    assert system.working.capacity == 8
    assert system.cad.capacity == 8


def test_system_save_load(tmp_path):
    system = MemorySystem()
    system.remember("cad", "part:flange", {"op": "extrude"})
    system.save(str(tmp_path))
    other = MemorySystem()
    other.load(str(tmp_path))
    assert other.recall("cad", "part:flange").content == {"op": "extrude"}


def test_system_summary():
    system = MemorySystem()
    system.remember("cad", "part:flange", {"op": "extrude"})
    summary = system.summary()
    assert len(summary["pools"]) == 8
    assert summary["total_slots"] == 1
