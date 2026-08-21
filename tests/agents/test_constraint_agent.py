"""tests/agents/test_constraint_agent.py
=======================================
Unit tests for the constraint role agent.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.constraint import ConstraintAgent


def _payload():
    return {
        "variables": [
            {"name": "x", "initial": 1.0, "lower": 0.0, "upper": 10.0},
            {"name": "y", "initial": 2.0, "lower": 0.0, "upper": 10.0},
        ],
        "constraints": [
            {
                "name": "x_equals_y",
                "terms": {"x": 1.0, "y": -1.0},
                "operator": "==",
                "rhs": 0.0,
            }
        ],
    }


def test_solve_feasible():
    agent = ConstraintAgent()
    result = agent.process(AgentRequest(role="constraint", action="solve", payload=_payload()))
    assert result.ok
    assert result.output["feasible"] is True


def test_check():
    agent = ConstraintAgent()
    result = agent.process(AgentRequest(role="constraint", action="check", payload=_payload()))
    assert result.ok


def test_infeasible_bounds():
    agent = ConstraintAgent()
    payload = {
        "variables": [
            {"name": "x", "initial": 1.0, "lower": 0.0, "upper": 1.0},
        ],
        "constraints": [{"name": "x_big", "terms": {"x": 1.0}, "operator": "==", "rhs": 10.0}],
    }
    result = agent.process(AgentRequest(role="constraint", action="solve", payload=payload))
    assert not result.ok
