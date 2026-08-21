"""tests/agents/test_base.py
===========================
Unit tests for cadgenesis.agents.base.
"""

from __future__ import annotations

import pytest

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult


class _DummyAgent(Agent):
    role = "dummy"
    actions = ("ping",)

    def process(self, request: AgentRequest) -> AgentResult:
        return AgentResult(
            role=self.role,
            action=request.action,
            ok=True,
            output={"pong": request.payload.get("n", 0)},
            task_id=request.task_id,
        )


def test_request_validation():
    with pytest.raises(ValueError):
        AgentRequest(role="", action="ping")
    with pytest.raises(ValueError):
        AgentRequest(role="x", action="")


def test_agent_can_handle():
    agent = _DummyAgent()
    assert agent.can_handle("ping")
    assert not agent.can_handle("nope")


def test_agent_handle_dispatch():
    agent = _DummyAgent()
    result = agent.handle(AgentRequest(role="dummy", action="ping", payload={"n": 3}))
    assert result.ok
    assert result.output == {"pong": 3}


def test_agent_rejects_unknown_action():
    agent = _DummyAgent()
    result = agent.handle(AgentRequest(role="dummy", action="nope"))
    assert not result.ok
    assert "cannot handle" in result.message


def test_agent_requires_role():
    with pytest.raises(ValueError):

        class _NoRole(Agent):
            actions = ("x",)

            def process(self, request: AgentRequest) -> AgentResult:
                return AgentResult(role="", action=request.action, ok=True)

        _NoRole()


def test_agent_describe():
    agent = _DummyAgent()
    assert agent.describe() == {"role": "dummy", "actions": ["ping"]}
