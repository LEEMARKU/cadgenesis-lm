"""tests/agents/test_geometry_agent.py
====================================
Unit tests for the geometry role agent.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.geometry import GeometryAgent

_BLOCK = {
    "kind": "box",
    "dims": {"length": 2.0, "width": 3.0, "height": 4.0},
    "name": "block",
}


def test_validate():
    agent = GeometryAgent()
    result = agent.process(AgentRequest(role="geometry", action="validate", payload=_BLOCK))
    assert result.ok
    assert result.output["valid"] is True


def test_validate_invalid():
    agent = GeometryAgent()
    result = agent.process(
        AgentRequest(role="geometry", action="validate", payload={"kind": "box", "dims": {}})
    )
    assert not result.ok


def test_volume():
    agent = GeometryAgent()
    result = agent.process(AgentRequest(role="geometry", action="volume", payload=_BLOCK))
    assert result.ok
    assert result.output["volume"] == 24.0


def test_aabb():
    agent = GeometryAgent()
    result = agent.process(AgentRequest(role="geometry", action="aabb", payload=_BLOCK))
    assert result.ok
    assert result.output["min"] == [-1.0, -1.5, -2.0]
    assert result.output["max"] == [1.0, 1.5, 2.0]


def test_overlap_and_fit():
    agent = GeometryAgent()
    a = {"kind": "box", "dims": {"length": 1, "width": 1, "height": 1}}
    b = {"kind": "box", "dims": {"length": 5, "width": 5, "height": 5}}
    b_shifted = dict(b, position=(10.0, 10.0, 10.0))
    overlap = agent.process(
        AgentRequest(
            role="geometry",
            action="overlap",
            payload={"a": a, "b": b_shifted},
        )
    )
    assert not overlap.output["overlaps"]
    fit = agent.process(AgentRequest(role="geometry", action="fit", payload={"a": a, "b": b}))
    assert fit.output["fits"] is True


def test_missing_fields():
    agent = GeometryAgent()
    result = agent.process(AgentRequest(role="geometry", action="volume", payload={}))
    assert not result.ok
