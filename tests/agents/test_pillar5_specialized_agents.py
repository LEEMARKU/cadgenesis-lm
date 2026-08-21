"""tests/agents/test_pillar5_specialized_agents.py
=================================================
Unit tests for the 10 Pillar 5 specialized agents.
"""

from __future__ import annotations

from cadgenesis.agents.base import AgentRequest
from cadgenesis.agents.cost import CostAgent
from cadgenesis.agents.debugging import DebuggingAgent
from cadgenesis.agents.documentation import DocumentationAgent
from cadgenesis.agents.learning import LearningAgent
from cadgenesis.agents.material import MaterialAgent
from cadgenesis.agents.memory import MemoryAgent
from cadgenesis.agents.monitoring import MonitoringAgent
from cadgenesis.agents.retrieval import RetrievalAgent
from cadgenesis.agents.safety import SafetyComplianceAgent
from cadgenesis.agents.user import UserInteractionAgent


def _call(agent, action, payload):
    return agent.handle(AgentRequest(agent.role, action, payload))


# -------------------------------------------------------------------- material


def test_material_agent_lookup():
    result = _call(MaterialAgent(), "lookup", {"material": "Al 6061-T6"})
    assert result.ok
    assert result.output["material"] == "Al 6061-T6"
    assert result.output["properties"]


def test_material_agent_select():
    result = _call(
        MaterialAgent(),
        "select",
        {"required": {"yield_min": 100, "cost_max": 10.0}},
    )
    assert result.ok
    assert result.output["matches"]


# ------------------------------------------------------------------------ cost


def test_cost_agent_estimate():
    result = _call(CostAgent(), "estimate", {"mass_kg": 2.0, "quantity": 10})
    assert result.ok
    assert result.output["total_cost_usd"] > 0
    assert result.output["unit_cost_usd"] > 0


# --------------------------------------------------------------- documentation


def test_documentation_agent_summarize():
    result = _call(DocumentationAgent(), "summarize", {"topic": "bracket design", "findings": "ok"})
    assert result.ok
    assert "bracket design" in result.output["markdown"]


# ----------------------------------------------------------------------- safety


def test_safety_agent_compliance():
    result = _call(SafetyComplianceAgent(), "compliance", {"rules": ["sharp_edge", "burr"]})
    assert result.ok
    assert "sharp_edge" in [r["rule"] for r in result.output["rules"]]


# ----------------------------------------------------------------------- memory


def test_memory_agent_remember_and_recall():
    agent = MemoryAgent()
    remembered = _call(agent, "remember", {"key": "k1", "content": "v1"})
    assert remembered.ok
    recalled = _call(agent, "recall", {"key": "k1"})
    assert recalled.ok
    assert recalled.output["content"] == "v1"


# -------------------------------------------------------------------- retrieval


def test_retrieval_agent_retrieve():
    memory = MemoryAgent().memory
    memory.remember("engineering", "lesson:fit", "parts must clear")
    result = _call(RetrievalAgent(memory=memory), "retrieve", {"query": "fit"})
    assert result.ok
    assert result.output["hits"]


def test_retrieval_agent_route():
    memory = MemoryAgent().memory
    memory.remember("engineering", "lesson:fit", "parts must clear")
    result = _call(RetrievalAgent(memory=memory), "route", {"query": "fit"})
    assert result.ok
    assert result.output["route"]


# ------------------------------------------------------------------------ user


def test_user_agent_preferences():
    agent = UserInteractionAgent()
    result = _call(agent, "set_preference", {"name": "units", "value": "metric"})
    assert result.ok
    result = _call(agent, "get_preference", {"name": "units"})
    assert result.output["value"] == "metric"
    result = _call(agent, "record_style", {"style": "minimalist"})
    assert result.ok


# ---------------------------------------------------------------------- learning


def test_learning_agent_experience():
    agent = LearningAgent()
    recorded = _call(agent, "record_experience", {"task": "clearance", "lesson": "keep 1mm gap"})
    assert recorded.ok
    recalled = _call(agent, "recall_lessons", {"query": "clearance"})
    assert recalled.ok


# --------------------------------------------------------------------- monitoring


def test_monitoring_agent_health():
    from cadgenesis.agents.health import AgentHealthMonitor

    monitor = AgentHealthMonitor()
    from cadgenesis.agents.material import MaterialAgent

    agent = MaterialAgent()
    monitor.register(agent)
    result = _call(MonitoringAgent(monitor=monitor), "health", {"agents": [agent]})
    assert result.ok
    assert "material" in result.output["summary"]["healthy"]


# --------------------------------------------------------------------- debugging


def test_debugging_agent_inspect():
    from cadgenesis.agents.base import AgentResult

    results = [
        AgentResult("a", "x", True, {}, "ok"),
        AgentResult("b", "y", False, {}, "KeyError: missing key"),
    ]
    result = _call(DebuggingAgent(), "inspect", {"results": results})
    assert result.ok
    assert result.output["failed"] == 1
    assert result.output["causes"]["KeyError"] == 1


def test_debugging_agent_suggest_fix():
    result = _call(DebuggingAgent(), "suggest_fix", {"message": "KeyError: boom"})
    assert result.ok
    assert result.output["suggestion"]


def test_unsupported_action_fails():
    result = _call(MaterialAgent(), "nope", {})
    assert not result.ok
