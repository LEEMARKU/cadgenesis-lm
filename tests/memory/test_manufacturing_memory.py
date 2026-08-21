"""tests/memory/test_manufacturing_memory.py
===========================================
Unit tests for the manufacturing memory pool.
"""

from __future__ import annotations

from cadgenesis.memory.manufacturing_memory import ManufacturingMemory


def test_process_limits():
    store = ManufacturingMemory(capacity=16)
    store.remember_process("milling", {"max_tool_diameter": 20})
    assert store.process_limits("milling")["max_tool_diameter"] == 20
    assert store.process_limits("missing") is None


def test_processes_list():
    store = ManufacturingMemory(capacity=16)
    store.remember_process("milling", {"x": 1})
    store.remember_process("turning", {"x": 2})
    assert set(store.processes()) == {"milling", "turning"}


def test_recall():
    store = ManufacturingMemory(capacity=16)
    store.remember_process("milling", {"max_tool_diameter": 20})
    assert store.recall("tool")[0].entry.key == "process:milling"
