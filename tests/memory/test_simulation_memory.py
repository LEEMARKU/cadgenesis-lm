"""tests/memory/test_simulation_memory.py
========================================
Unit tests for the simulation memory pool.
"""

from __future__ import annotations

from cadgenesis.memory.simulation_memory import SimulationMemory


def test_remember_result():
    store = SimulationMemory(capacity=16)
    store.remember_result("run:1", {"safety_factor": 1.5}, analysis_type="structural")
    assert store.by_analysis_type("structural")[0].key == "run:1"


def test_recall_filtered():
    store = SimulationMemory(capacity=16)
    store.remember_result("run:1", {"safety_factor": 1.5}, analysis_type="structural")
    store.remember_result("run:2", {"velocity": 12.0}, analysis_type="fluid")
    assert store.recall("safety", analysis_type="structural")[0].entry.key == "run:1"
    assert all(
        hit.entry.metadata["analysis_type"] == "fluid"
        for hit in store.recall("safety", analysis_type="fluid")
    )
