"""tests/agents/test_pillar5_consensus.py
========================================
Unit tests for the Pillar 5 consensus extensions (quorum, veto, tie-break,
fallback, decision trace).
"""

from __future__ import annotations

from cadgenesis.agents.consensus import AgentOpinion, ConsensusEngine


def _opts(engine, *options):
    for i, option in enumerate(options):
        engine.record(AgentOpinion(f"agent{i}", option))


def test_quorum_required():
    engine = ConsensusEngine(quorum=3, fallback="undecided")
    engine.record(AgentOpinion("a", "yes"))
    assert not engine.has_quorum()
    assert engine.decision() == "undecided"
    engine.record(AgentOpinion("b", "yes"))
    engine.record(AgentOpinion("c", "no"))
    assert engine.has_quorum()
    assert engine.decision() == "yes"


def test_veto_agents_block():
    engine = ConsensusEngine(quorum=2, veto_agents=("guard",), fallback="blocked")
    engine.record(AgentOpinion("guard", "yes"))
    engine.record(AgentOpinion("peer", "yes"))
    assert engine.vetoed()
    assert engine.decision() == "blocked"


def test_tie_break_weighted():
    engine = ConsensusEngine(tie_break="weighted")
    engine.record(AgentOpinion("a", "x", weight=1.0, confidence=1.0))
    engine.record(AgentOpinion("b", "y", weight=2.0, confidence=1.0))
    engine.record(AgentOpinion("c", "y", weight=1.0, confidence=1.0))
    assert engine.majority() == "y"


def test_tie_break_first_preserves_legacy():
    engine = ConsensusEngine(tie_break="first")
    engine.record(AgentOpinion("a", "x"))
    engine.record(AgentOpinion("b", "y"))
    assert engine.majority() == "x"


def test_decision_trace():
    engine = ConsensusEngine(quorum=1)
    engine.record(AgentOpinion("a", "yes"))
    decision = engine.decision()
    assert decision == "yes"
    assert engine.trace[-1]["decision"] == "yes"
    assert engine.trace[-1]["method"] == "majority"


def test_full_summary_keys():
    engine = ConsensusEngine(quorum=1)
    engine.record(AgentOpinion("a", "yes"))
    summary = engine.full_summary()
    for key in ("quorum", "has_quorum", "vetoed", "tie_break", "decision", "trace"):
        assert key in summary
