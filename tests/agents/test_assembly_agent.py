"""tests/agents/test_assembly_agent.py
====================================
Unit tests for the assembly role agent.
"""

from __future__ import annotations

from cadgenesis.agents.assembly import AssemblyAgent
from cadgenesis.agents.base import AgentRequest


def _payload(shifted=False):
    a = {"kind": "box", "dims": {"length": 1, "width": 1, "height": 1}, "name": "a"}
    b = {"kind": "box", "dims": {"length": 5, "width": 5, "height": 5}, "name": "b"}
    if shifted:
        b["position"] = (10.0, 10.0, 10.0)
    return {"a": a, "b": b}


def test_clearance_ok():
    agent = AssemblyAgent()
    result = agent.process(
        AgentRequest(role="assembly", action="check_clearance", payload=_payload(shifted=True))
    )
    assert result.ok


def test_clearance_violated():
    agent = AssemblyAgent()
    result = agent.process(
        AgentRequest(
            role="assembly",
            action="check_clearance",
            payload={**_payload(), "gap": 100.0},
        )
    )
    assert not result.ok
    assert "clearance" in result.message


def test_mate():
    agent = AssemblyAgent()
    # Separate parts -> interference-free mate.
    result = agent.process(
        AgentRequest(role="assembly", action="check_mate", payload=_payload(shifted=True))
    )
    assert result.ok


def test_mate_overlap_fails():
    agent = AssemblyAgent()
    # Overlapping AABBs -> mate is invalid.
    result = agent.process(AgentRequest(role="assembly", action="check_mate", payload=_payload()))
    assert not result.ok
