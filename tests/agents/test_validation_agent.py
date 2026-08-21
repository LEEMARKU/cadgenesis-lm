"""tests/agents/test_validation_agent.py
=======================================
Unit tests for the validation role agent.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.validation import ValidationAgent


def test_validate_empty_context():
    agent = ValidationAgent()
    result = agent.process(
        AgentRequest(role="validation", action="validate", payload={"context": {}})
    )
    assert result.ok  # no checks configured -> passes
    assert result.output["passed"] is True


def test_validate_missing_context():
    agent = ValidationAgent()
    result = agent.process(AgentRequest(role="validation", action="validate", payload={}))
    assert not result.ok


def test_report():
    agent = ValidationAgent()
    result = agent.process(
        AgentRequest(role="validation", action="report", payload={"context": {}})
    )
    assert result.ok
    assert result.output["passed"] is True
    assert result.output["total"] >= 0


def test_validate_with_custom_check():
    from cadgenesis.reasoning.validator import CheckResult

    validator = ValidationAgent().validator

    def custom(context):
        return [CheckResult(category="custom", name="size", passed=context["size"] > 0)]

    validator.add_check(custom)
    agent = ValidationAgent(validator=validator)
    result = agent.process(
        AgentRequest(
            role="validation",
            action="validate",
            payload={"context": {"size": 5}},
        )
    )
    assert result.ok
