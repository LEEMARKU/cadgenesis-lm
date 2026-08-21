"""tests/agents/test_coordinator.py
==================================
Unit tests for cadgenesis.agents.coordinator.
"""

from __future__ import annotations

from cadgenesis.agents.coordinator import AgentCoordinator
from cadgenesis.agents.optimization import OptimizationAgent
from cadgenesis.agents.planner import PlannerAgent


def _build_coordinator() -> AgentCoordinator:
    return AgentCoordinator(agents=[PlannerAgent(), OptimizationAgent()])


def test_register_and_roles():
    coordinator = _build_coordinator()
    assert set(coordinator.roles) == {"planner", "optimization"}
    assert coordinator.agent("planner") is not None
    assert coordinator.agent("missing") is None


def test_dispatch_unknown_role():
    coordinator = AgentCoordinator()
    result = coordinator.dispatch_action("ghost", "anything")
    assert not result.ok
    assert "no agent registered" in result.message


def test_dispatch_unknown_action():
    coordinator = _build_coordinator()
    result = coordinator.dispatch_action("planner", "bogus")
    assert not result.ok


def test_dispatch_planner():
    coordinator = _build_coordinator()
    result = coordinator.dispatch_action("planner", "create_plan", {"goal": "box"})
    assert result.ok
    assert result.output["steps"] >= 1


def test_share_and_publish():
    coordinator = _build_coordinator()
    coordinator.share("design", {"part": "flange"})
    assert coordinator.memory.get("design") == {"part": "flange"}
    message = coordinator.publish("topic.x", {"k": 1})
    assert message.payload == {"k": 1}


def test_ask_consensus():
    coordinator = AgentCoordinator(agents=[OptimizationAgent(), OptimizationAgent(target_cost=5.0)])
    summary = coordinator.ask_consensus(
        "optimization",
        "optimize",
        {"objective": "mass", "params": {"current": 5.0, "target": 5.0}},
        options=[True, False],
    )
    assert summary["count"] >= 1
    assert summary["majority"] is True


def test_run_batch():
    coordinator = _build_coordinator()
    coordinator.submit("planner", "create_plan", {"goal": "box"}, task_id="t1")
    results = coordinator.run_batch()
    assert results
    assert results[0].ok
    assert coordinator.scheduler.progress()["completed"] == 1


def test_run_all_dependencies():
    coordinator = _build_coordinator()
    coordinator.submit("planner", "create_plan", {"goal": "box"}, task_id="t2", depends_on=["t1"])
    coordinator.submit("planner", "create_plan", {"goal": "box"}, task_id="t1")
    results = coordinator.run_all()
    assert len(results) == 2
    assert coordinator.scheduler.progress()["completed"] == 2


def test_summary():
    coordinator = _build_coordinator()
    summary = coordinator.summary()
    assert "agents" in summary
    assert "scheduler" in summary
