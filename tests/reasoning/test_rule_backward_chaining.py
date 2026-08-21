"""Tests for Pillar 7 backward chaining and rule versioning."""

from __future__ import annotations

import pytest

from cadgenesis.reasoning import (
    Proof,
    Rule,
    RuleEngine,
    make_rule,
)


def _chains_engine() -> RuleEngine:
    engine = RuleEngine()
    engine.add_rules(
        [
            make_rule(
                "material_known",
                lambda ctx: "material" in ctx,
                severity="info",
                meta={"concludes": "material_known"},
            ),
            make_rule(
                "wall_ok",
                lambda ctx: ctx.get("min_wall", 0.0) >= 0.8,
                severity="error",
                meta={"concludes": "wall_ok"},
            ),
            make_rule(
                "draft_ok",
                lambda ctx: ctx.get("draft_angle", 0.0) >= 1.0,
                severity="warning",
                meta={"concludes": "draft_ok", "requires": ["wall_ok"]},
            ),
            make_rule(
                "design_ok",
                lambda ctx: bool(ctx.get("wall_ok")),
                severity="info",
                meta={"concludes": "design_ok", "requires": ["wall_ok", "draft_ok"]},
            ),
        ]
    )
    return engine


def test_prove_fact_in_context() -> None:
    engine = _chains_engine()
    proof = engine.prove("material", {"material": "steel"})
    assert proof.established
    assert proof.depth == 0
    assert proof.steps == []


def test_prove_via_rule_chain() -> None:
    engine = _chains_engine()
    context = {"min_wall": 1.0, "draft_angle": 2.0}
    proof = engine.prove("design_ok", context)
    assert proof.established
    assert "wall_ok" in proof.steps
    assert "draft_ok" in proof.steps


def test_prove_fails_when_rule_fires_but_context_stays_false() -> None:
    engine = _chains_engine()
    proof = engine.prove("design_ok", {"min_wall": 0.2, "draft_angle": 2.0})
    assert not proof.established


def test_prove_context_not_mutated() -> None:
    engine = _chains_engine()
    context = {"min_wall": 1.0, "draft_angle": 2.0}
    before = dict(context)
    engine.prove("design_ok", context)
    assert context == before


def test_prove_respects_depth_limit() -> None:
    engine = _chains_engine()
    with pytest.raises(ValueError):
        engine.prove("design_ok", {"min_wall": 1.0, "draft_angle": 2.0}, depth_limit=0)
    proof = engine.prove("design_ok", {"min_wall": 1.0, "draft_angle": 2.0}, depth_limit=1)
    assert not proof.established


def test_prove_all() -> None:
    engine = _chains_engine()
    proofs = engine.prove_all(["wall_ok", "design_ok"], {"min_wall": 1.0, "draft_angle": 2.0})
    assert isinstance(proofs["wall_ok"], Proof)
    assert proofs["wall_ok"].established
    assert proofs["design_ok"].established


def test_proof_summary() -> None:
    engine = _chains_engine()
    proof = engine.prove("wall_ok", {"min_wall": 1.0})
    summary = proof.summary()
    assert summary["goal"] == "wall_ok"
    assert summary["established"] is True
    assert isinstance(summary["trace"], list)


def test_rule_version_default() -> None:
    rule = make_rule("r", lambda c: True)
    assert rule.version == "1.0.0"


def test_rule_version_validation() -> None:
    with pytest.raises(ValueError):
        Rule("r", lambda c: True, version="")
    with pytest.raises(ValueError):
        Rule("r", lambda c: True, version=3)


def test_rule_requires_and_concludes() -> None:
    rule = make_rule(
        "r",
        lambda c: True,
        meta={"concludes": "goal", "requires": ["a", "b"]},
    )
    assert rule.concludes() == "goal"
    assert rule.requires() == ["a", "b"]


def test_versioned_snapshot_and_diff() -> None:
    a = RuleEngine([make_rule("r1", lambda c: True, version="1.0.0")])
    b = RuleEngine([make_rule("r1", lambda c: True, version="2.0.0")])
    b.add_rules([make_rule("r2", lambda c: True)])
    snapshot = b.snapshot()
    assert snapshot["total"] == 2
    diff = a.diff(b)
    assert diff["changed"] == ["r1"]
    assert diff["added"] == ["r2"]
    assert diff["removed"] == []


def test_by_version() -> None:
    engine = RuleEngine(
        [
            make_rule("v1rule", lambda c: True, version="1.0.0"),
            make_rule("v2rule", lambda c: True, version="2.0.0"),
        ]
    )
    assert len(engine.by_version("2.0.0")) == 1
    assert engine.by_version("2.0.0")[0].name == "v2rule"
