"""tests/continual_learning/test_replay_buffer.py
================================================
Unit tests for the Pillar 6 replay buffer (MemorySystem substrate).
"""

from __future__ import annotations

import random

import pytest

from cadgenesis.continual_learning.replay_buffer import ReplayBuffer
from cadgenesis.memory.memory_system import MemorySystem


def test_store_and_len():
    buffer = ReplayBuffer(MemorySystem())
    buffer.store("experience one")
    buffer.store("experience two", importance=0.9)
    assert len(buffer) == 2


def test_store_many():
    buffer = ReplayBuffer(MemorySystem())
    keys = buffer.store_many(["a", "b", "c"])
    assert len(keys) == 3
    assert len(buffer) == 3


def test_uniform_sample_within_bounds():
    buffer = ReplayBuffer(MemorySystem())
    buffer.store_many(["a", "b", "c", "d", "e"])
    samples = buffer.sample(batch_size=3, strategy="uniform")
    assert len(samples) == 3
    assert all(isinstance(s.content, str) for s in samples)


def test_importance_sample():
    random.seed(0)  # deterministic: weighted draw of 2 from {0.1, 10.0}
    buffer = ReplayBuffer(MemorySystem())
    buffer.store("low", importance=0.1, key="low")
    buffer.store("high", importance=10.0, key="high")
    samples = buffer.sample(batch_size=10, strategy="importance")
    keys = {s.key for s in samples}
    assert "high" in keys  # overwhelmingly likely with 10:0.1 weights


def test_sample_empty():
    buffer = ReplayBuffer(MemorySystem())
    assert buffer.sample(batch_size=8) == []


def test_sample_bad_strategy():
    buffer = ReplayBuffer(MemorySystem())
    buffer.store("x")
    with pytest.raises(ValueError):
        buffer.sample(strategy="bogus")


def test_recall_by_query():
    buffer = ReplayBuffer(MemorySystem())
    buffer.store("machining tolerance for shafts")
    buffer.store("user preference about colors")
    hits = buffer.recall("tolerance", top_k=8)
    assert len(hits) == 1
    assert "machining" in hits[0].content


def test_clear():
    buffer = ReplayBuffer(MemorySystem())
    buffer.store("x")
    buffer.clear()
    assert len(buffer) == 0
