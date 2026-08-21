"""tests/agents/test_simulation_agent.py
=======================================
Unit tests for the simulation role agent.
"""

from __future__ import annotations

import pytest

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.simulation import SimulationAgent


def test_check_safety_pass():
    agent = SimulationAgent()
    result = agent.process(
        AgentRequest(
            role="simulation",
            action="check_safety",
            payload={"safety_factor": 2.0, "required_safety_factor": 1.5},
        )
    )
    assert result.ok
    assert result.output["margin"] == 0.5


def test_check_safety_fail():
    agent = SimulationAgent()
    result = agent.process(
        AgentRequest(
            role="simulation",
            action="check_safety",
            payload={"safety_factor": 1.2, "required_safety_factor": 1.5},
        )
    )
    assert not result.ok


def test_check_load_case():
    agent = SimulationAgent()
    result = agent.process(
        AgentRequest(
            role="simulation",
            action="check_load_case",
            payload={"loads": [{"magnitude": 100.0}, {"magnitude": 200.0}]},
        )
    )
    assert result.ok
    assert result.output["max_magnitude"] == 200.0


def test_check_load_case_empty():
    agent = SimulationAgent()
    result = agent.process(
        AgentRequest(role="simulation", action="check_load_case", payload={"loads": []})
    )
    assert not result.ok


def test_invalid_default_safety_factor():
    with pytest.raises(ValueError):
        SimulationAgent(default_safety_factor=0.0)
