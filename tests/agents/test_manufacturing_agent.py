"""tests/agents/test_manufacturing_agent.py
==========================================
Unit tests for the manufacturing role agent.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.manufacturing import ManufacturingAgent


def _good_part():
    return {
        "processes": ["machining"],
        "min_wall_thickness": 3.0,
        "min_feature_size": 2.0,
        "max_aspect_ratio": 4.0,
    }


def test_assess():
    agent = ManufacturingAgent()
    result = agent.process(
        AgentRequest(role="manufacturing", action="assess", payload={"part": _good_part()})
    )
    assert result.ok
    assert result.output["passed"] is True


def test_assess_missing_part():
    agent = ManufacturingAgent()
    result = agent.process(AgentRequest(role="manufacturing", action="assess", payload={}))
    assert not result.ok


def test_check_process():
    agent = ManufacturingAgent()
    result = agent.process(
        AgentRequest(
            role="manufacturing",
            action="check_process",
            payload={"part": _good_part(), "process": "machining"},
        )
    )
    assert result.ok


def test_check_process_unknown():
    agent = ManufacturingAgent()
    result = agent.process(
        AgentRequest(
            role="manufacturing",
            action="check_process",
            payload={"part": _good_part(), "process": "welding"},
        )
    )
    assert not result.ok
