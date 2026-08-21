"""tests/agents/test_planner_agent.py
===================================
Unit tests for the planner role agent.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.planner import PlannerAgent


def test_create_plan():
    agent = PlannerAgent()
    result = agent.process(
        AgentRequest(role="planner", action="create_plan", payload={"goal": "box"})
    )
    assert result.ok
    assert result.output["goal"] == "box"
    assert result.output["steps"] == 4


def test_create_plan_missing_goal():
    agent = PlannerAgent()
    result = agent.process(AgentRequest(role="planner", action="create_plan", payload={}))
    assert not result.ok


def test_refine_plan():
    agent = PlannerAgent()
    plan = agent.planner.create_plan("box").to_dict()
    result = agent.process(
        AgentRequest(role="planner", action="refine_plan", payload={"plan": plan})
    )
    assert result.ok
    assert "plan" in result.output


def test_unsupported_action():
    agent = PlannerAgent()
    result = agent.process(AgentRequest(role="planner", action="nope", payload={}))
    assert not result.ok
