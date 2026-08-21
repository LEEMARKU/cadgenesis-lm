"""tests/agents/test_optimization_agent.py
=========================================
Unit tests for the optimization role agent.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.optimization import OptimizationAgent


def test_at_target():
    agent = OptimizationAgent()
    result = agent.process(
        AgentRequest(
            role="optimization",
            action="optimize",
            payload={"objective": "mass", "params": {"current": 5.0, "target": 5.0}},
        )
    )
    assert result.ok
    assert result.output["score"] == 1.0


def test_off_target():
    agent = OptimizationAgent()
    result = agent.process(
        AgentRequest(
            role="optimization",
            action="optimize",
            payload={"objective": "mass", "params": {"current": 10.0, "target": 5.0}},
        )
    )
    assert not result.ok
    assert "decrease mass" in result.output["recommendation"]


def test_suggest():
    agent = OptimizationAgent()
    result = agent.process(
        AgentRequest(
            role="optimization",
            action="suggest",
            payload={"objective": "cost", "params": {"current": 2.0, "target": 3.0}},
        )
    )
    assert result.ok is False
    assert "increase cost" in result.output["recommendation"]


def test_missing_params():
    agent = OptimizationAgent()
    result = agent.process(
        AgentRequest(role="optimization", action="optimize", payload={"objective": "mass"})
    )
    assert not result.ok


def test_target_cost():
    agent = OptimizationAgent(target_cost=5.0)
    result = agent.process(
        AgentRequest(
            role="optimization",
            action="optimize",
            payload={"objective": "mass", "params": {"current": 5.0, "target": 5.0}},
        )
    )
    assert result.ok
    assert 0.0 <= result.output["score"] <= 1.0
