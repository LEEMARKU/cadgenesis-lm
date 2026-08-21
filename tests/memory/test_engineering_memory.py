"""tests/memory/test_engineering_memory.py
=========================================
Unit tests for the engineering memory pool.
"""

from __future__ import annotations

from cadgenesis.memory.engineering_memory import EngineeringMemory


def test_remember_standard():
    store = EngineeringMemory(capacity=16)
    store.remember_standard("ISO-2768", "general tolerances")
    assert store.standard("ISO-2768") == "general tolerances"
    assert store.standard("missing") is None


def test_recall():
    store = EngineeringMemory(capacity=16)
    store.remember_standard("ISO-2768", "general tolerances for flanges")
    assert store.recall("tolerances")[0].entry.key == "standard:ISO-2768"


def test_guidelines():
    store = EngineeringMemory(capacity=16)
    store.add("g1", "wall thickness guideline", metadata={"kind": "guideline"})
    assert store.guidelines()[0].key == "g1"
