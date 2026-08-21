"""tests/memory/test_user_memory.py
==================================
Unit tests for the user memory pool.
"""

from __future__ import annotations

from cadgenesis.memory.user_memory import UserMemory


def test_preferences():
    store = UserMemory(capacity=16)
    store.set_preference("units", "metric")
    assert store.get_preference("units") == "metric"
    assert store.get_preference("missing", "mm") == "mm"


def test_preferences_dict():
    store = UserMemory(capacity=16)
    store.set_preference("units", "metric")
    store.set_preference("tolerance", 0.1)
    assert store.preferences() == {"units": "metric", "tolerance": 0.1}


def test_style():
    store = UserMemory(capacity=16)
    store.record_style({"name": "minimal", "tolerance": 0.1})
    assert store.style("minimal")["tolerance"] == 0.1
    assert store.style("missing") is None
