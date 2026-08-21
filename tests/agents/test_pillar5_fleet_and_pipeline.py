"""tests/agents/test_pillar5_fleet_and_pipeline.py
=================================================
Unit tests for the 18-agent fleet, the task-planning pipeline and the
AgentPlatform orchestrator.
"""

from __future__ import annotations

from cadgenesis.agents.fleet import FLEET_ROLES, build_fleet, create_fleet_registry
from cadgenesis.agents.orchestrator import AgentPlatform
from cadgenesis.agents.pipeline import (
    IntentAnalyser,
    ResultAggregator,
    TaskPlanningPipeline,
    TaskValidator,
)


def test_fleet_has_all_18_roles():
    registry = create_fleet_registry()
    roles = sorted(a.role for a in registry.agents)
    assert sorted(roles) == sorted(FLEET_ROLES)
    assert len(roles) == 18


def test_fleet_mixes_legacy_and_new():
    registry = create_fleet_registry()
    assert registry.get("planner") is not None
    assert registry.get("geometry") is not None
    assert registry.get("cost") is not None
    assert registry.get("debugging") is not None


def test_build_fleet_registers_into_provided_registry():
    from cadgenesis.agents.registry import AgentRegistry

    registry = AgentRegistry()
    agents = build_fleet(registry=registry)
    assert len(agents) == 18
    assert len(registry) == 18


# -------------------------------------------------------------------- pipeline


def test_intent_analyser_detects_roles():
    result = IntentAnalyser().analyse("validate the cost of the assembly")
    assert "validation" in result["detected_roles"]
    assert "cost" in result["detected_roles"]


def test_pipeline_runs_end_to_end():
    registry = create_fleet_registry()
    pipeline = TaskPlanningPipeline(registry)
    report = pipeline.run("plan and validate a cost-optimized assembly")
    try:
        assert report.validation["passed_count"] >= 1
        assert report.aggregated["ok"] >= 1
        assert report.duration_s >= 0.0
    finally:
        pipeline.shutdown()


def test_pipeline_with_decomposition():
    registry = create_fleet_registry()
    pipeline = TaskPlanningPipeline(registry)
    report = pipeline.run("plan and validate a cost-optimized assembly", decompose=True)
    try:
        assert len(report.tasks) >= 1
    finally:
        pipeline.shutdown()


def test_task_validator_flags_failures():
    from cadgenesis.agents.base import AgentResult

    validator = TaskValidator()
    validation = validator.validate(
        [AgentResult("a", "x", True, {}, "ok"), AgentResult("b", "y", False, {}, "bad")]
    )
    assert not validation["passed"]
    assert validation["failed_count"] == 1


def test_result_aggregator():
    from cadgenesis.agents.base import AgentResult

    aggregator = ResultAggregator()
    aggregated = aggregator.aggregate([AgentResult("a", "x", True, {"v": 1}, "done")])
    assert aggregated["ok"] == 1
    assert "a:x" in aggregated["outputs"]


# ------------------------------------------------------------------ orchestrator


def test_platform_load_fleet_and_dispatch():
    platform = AgentPlatform()
    platform.load_fleet()
    result = platform.dispatch("material", "lookup", {"material": "Al 6061-T6"})
    assert result.ok
    assert result.output["material"] == "Al 6061-T6"


def test_platform_unknown_role_fails():
    platform = AgentPlatform()
    result = platform.dispatch("ghost", "lookup", {})
    assert not result.ok


def test_platform_share_and_publish():
    platform = AgentPlatform()
    platform.share("working", "wip", {"stage": 1})
    assert platform.memory.get("working", "wip") == {"stage": 1}
    event = platform.publish("design.updated", {"id": 5})
    assert event.topic == "design.updated"
    platform.shutdown()


def test_platform_ask_returns_summary():
    platform = AgentPlatform()
    platform.load_fleet()
    summary = platform.ask("is the design acceptable?")
    assert "decision" in summary
    platform.shutdown()


def test_platform_health_and_stats():
    platform = AgentPlatform()
    platform.load_fleet()
    summary = platform.health_summary()
    assert summary["total"] == 18
    assert platform.stats()["agents"] == 18
    platform.shutdown()


def test_platform_submit_pipeline():
    platform = AgentPlatform()
    platform.load_fleet()
    report = platform.submit_pipeline("plan and validate a cost-optimized assembly")
    assert "validation" in report
    platform.shutdown()
