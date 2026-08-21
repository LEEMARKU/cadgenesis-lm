"""Tests for Pillar 7 hybrid neuro-symbolic pipeline and neural engine forward."""

from __future__ import annotations

import pytest
import torch

from cadgenesis.reasoning import (
    Constraint,
    HybridReasoningPipeline,
    NeuroSymbolicReasoningEngine,
    Primitive,
    RuleEngine,
    Variable,
    make_rule,
)
from cadgenesis.reasoning.standards import build_standards_graph


def _rules() -> RuleEngine:
    engine = RuleEngine()
    engine.add_rules(
        [
            make_rule(
                "wall_too_thin",
                lambda ctx: ctx.get("min_wall", 0.0) < 0.8,
                severity="error",
            ),
            make_rule(
                "draft_too_low",
                lambda ctx: ctx.get("draft_angle", 0.0) < 1.0,
                severity="warning",
            ),
        ]
    )
    return engine


def _pipeline(**kwargs) -> HybridReasoningPipeline:
    return HybridReasoningPipeline(rule_engine=_rules(), **kwargs)


def _good_context() -> dict:
    return {
        "id": "part-A",
        "query": "ISO 286",
        "min_wall": 1.0,
        "draft_angle": 2.0,
        "constraint_variables": [Variable("w", initial=2.0, lower=0.0, upper=5.0)],
        "constraints": [Constraint("c1", {"w": 1.0}, "==", 2.0)],
        "primitives": [Primitive("box", {"length": 1, "width": 1, "height": 1})],
        "part": {"processes": ["machining"], "min_wall_thickness": 1.0},
    }


def test_pipeline_passes_good_context() -> None:
    report = _pipeline().reason(_good_context())
    assert report.passed
    assert report.score >= 0.5
    assert "rules" in report.stage_names()
    assert "constraints" in report.stage_names()
    assert "geometry" in report.stage_names()
    assert "manufacturing" in report.stage_names()


def test_pipeline_fails_bad_rule() -> None:
    context = _good_context()
    context["min_wall"] = 0.1
    report = _pipeline().reason(context)
    assert not report.passed
    rules = report.stage("rules")
    assert rules is not None and not rules.passed


def test_pipeline_fails_constraints() -> None:
    context = _good_context()
    context["constraints"] = [
        Constraint("c1", {"w": 1.0}, "==", 2.0),
        Constraint("c2", {"w": 1.0}, ">=", 9.0),
    ]
    report = _pipeline().reason(context)
    constraints = report.stage("constraints")
    assert constraints is not None and not constraints.passed


def test_pipeline_fails_geometry_interference() -> None:
    context = _good_context()
    context["primitives"] = [
        Primitive("box", {"length": 10, "width": 10, "height": 10}, position=(0, 0, 0)),
        Primitive("box", {"length": 10, "width": 10, "height": 10}, position=(5, 5, 5)),
    ]
    report = _pipeline().reason(context)
    geometry = report.stage("geometry")
    assert geometry is not None and not geometry.passed


def test_pipeline_fails_manufacturing() -> None:
    context = _good_context()
    context["part"] = {"processes": ["machining"], "min_wall_thickness": 0.1}
    report = _pipeline().reason(context)
    manufacturing = report.stage("manufacturing")
    assert manufacturing is not None and not manufacturing.passed


def test_pipeline_knowledge_stage_with_required() -> None:
    graph = build_standards_graph()
    report = _pipeline(knowledge_graph=graph).reason(
        {**_good_context(), "required_knowledge": ["ISO 286-1", "ISO 261"]}
    )
    knowledge = report.stage("knowledge")
    assert knowledge is not None and knowledge.passed


def test_pipeline_knowledge_fails_missing() -> None:
    graph = build_standards_graph()
    report = _pipeline(knowledge_graph=graph).reason(
        {**_good_context(), "required_knowledge": ["ISO 99999"]}
    )
    knowledge = report.stage("knowledge")
    assert knowledge is not None and not knowledge.passed


def test_neural_stage_integrated() -> None:
    engine = NeuroSymbolicReasoningEngine(d_model=16)
    report = _pipeline(neural_engine=engine).reason(
        _good_context(),
        neural_hidden=torch.randn(1, 4, 16),
    )
    assert "neural" in report.stage_names()


def test_neural_refinement_kicks_in() -> None:
    engine = NeuroSymbolicReasoningEngine(d_model=16)
    engine.constraint_evaluator.bias.data.fill_(-0.15)  # soft neural miss ~0.46
    report = _pipeline(neural_engine=engine, threshold=0.5).reason(
        _good_context(), neural_hidden=torch.zeros(1, 4, 16)
    )
    assert report.refined
    assert "refinement" in report.stage_names()
    assert report.score >= 0.5
    assert report.passed


def test_custom_stage_registration() -> None:
    pipeline = _pipeline()
    pipeline.add_stage("material_ok", lambda ctx: ctx.get("material") == "steel")
    report = pipeline.reason(_good_context())
    assert "material_ok" in report.stage_names()
    bad = pipeline.reason({**_good_context(), "material": "wood"})
    material = bad.stage("material_ok")
    assert material is not None and not material.passed


def test_all_custom_stages_fail_blocks_decision() -> None:
    pipeline = _pipeline()
    pipeline.add_stage("material_ok", lambda ctx: ctx.get("material") == "steel")
    pipeline.add_stage("verified", lambda ctx: ctx.get("verified") is True)
    pipeline.add_stage("approved", lambda ctx: ctx.get("approved") is True)
    context = _good_context()
    context["material"] = "wood"
    report = pipeline.reason(context)
    assert not report.passed


def test_report_explain_and_summary() -> None:
    report = _pipeline().reason(_good_context())
    text = report.explain()
    assert "part-A" in text
    assert "PASS" in text
    summary = report.summary()
    assert summary["passed"] is True


def test_neural_engine_forward() -> None:
    engine = NeuroSymbolicReasoningEngine(d_model=16)
    facts = torch.randn(1, 4, 16)
    state = torch.randn(1, 4, 16)
    corrected, scores = engine.forward(facts, state)
    assert corrected.shape == (1, 4, 16)
    assert scores.shape == (1, 4, 1)
    assert bool((scores >= 0).all() and (scores <= 1).all())


def test_neural_engine_forward_rejects_non_tensor() -> None:
    engine = NeuroSymbolicReasoningEngine(d_model=16)
    with pytest.raises(TypeError):
        engine.forward({"facts": 1}, torch.randn(1, 4, 16))


def test_neural_engine_evaluate_constraints_unchanged() -> None:
    engine = NeuroSymbolicReasoningEngine(d_model=16)
    hidden = torch.randn(1, 4, 16)
    scores, corrected = engine.evaluate_constraints(hidden)
    assert scores.shape == (1, 4, 1)
    assert corrected.shape == (1, 4, 16)
