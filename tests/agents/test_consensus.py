"""tests/agents/test_consensus.py
===============================
Unit tests for cadgenesis.agents.consensus.
"""

from __future__ import annotations

import pytest

from cadgenesis.agents.consensus import AgentOpinion, ConsensusEngine


def _engine() -> ConsensusEngine:
    engine = ConsensusEngine()
    engine.record_many(
        [
            AgentOpinion("a", "steel", weight=1.0, confidence=0.9),
            AgentOpinion("b", "steel", weight=1.0, confidence=0.8),
            AgentOpinion("c", "aluminum", weight=1.0, confidence=0.6),
        ]
    )
    return engine


def test_majority():
    assert _engine().majority() == "steel"


def test_weighted_majority():
    engine = ConsensusEngine()
    engine.record(AgentOpinion("a", "steel", weight=5.0, confidence=1.0))
    engine.record(AgentOpinion("b", "aluminum", weight=1.0, confidence=1.0))
    assert engine.weighted_majority() == "steel"


def test_mean():
    engine = ConsensusEngine()
    engine.record(AgentOpinion("a", 1.0))
    engine.record(AgentOpinion("b", 3.0))
    assert engine.mean() == 2.0


def test_mean_non_numeric_returns_none():
    engine = ConsensusEngine()
    engine.record(AgentOpinion("a", "steel"))
    assert engine.mean() is None


def test_is_unanimous():
    engine = ConsensusEngine()
    engine.record(AgentOpinion("a", "x"))
    engine.record(AgentOpinion("b", "x"))
    assert engine.is_unanimous()
    engine.record(AgentOpinion("c", "y"))
    assert not engine.is_unanimous()


def test_empty_engine():
    engine = ConsensusEngine()
    assert engine.majority() is None
    assert engine.mean() is None
    assert not engine.is_unanimous()
    assert engine.confidence() == 0.0


def test_requires_agent_name():
    engine = ConsensusEngine()
    with pytest.raises(ValueError):
        engine.record(AgentOpinion("", "x"))


def test_clear_and_count():
    engine = _engine()
    assert engine.count == 3
    engine.clear()
    assert engine.count == 0


def test_summary():
    summary = _engine().summary()
    assert summary["count"] == 3
    assert summary["majority"] == "steel"
