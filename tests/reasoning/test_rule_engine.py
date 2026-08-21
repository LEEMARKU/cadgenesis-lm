"""tests/reasoning/test_rule_engine.py
====================================
Unit tests for cadgenesis.reasoning.rule_engine.
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.rule_engine import (
    Rule,
    RuleEngine,
    RuleResult,
    make_rule,
)


@pytest.fixture
def engine() -> RuleEngine:
    rules = [
        make_rule(
            "wall_ok",
            lambda ctx: ctx.get("wall_thickness", 0) >= 0.8,
            description="wall meets minimum",
            severity="info",
        ),
        make_rule(
            "wall_too_thin",
            lambda ctx: ctx.get("wall_thickness", 0) < 0.8,
            description="wall below minimum",
            severity="error",
            meta={"recommendation": "Increase wall thickness."},
        ),
    ]
    return RuleEngine(rules)


class TestRule:
    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            Rule(name="", condition=lambda c: True)

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError):
            Rule(name="x", condition=lambda c: True, severity="fatal")

    def test_non_callable_condition_rejected(self):
        with pytest.raises(TypeError):
            Rule(name="x", condition="yes")


class TestRuleEngine:
    def test_add_and_lookup(self, engine):
        assert engine.get_rule("wall_ok") is not None
        assert engine.get_rule("missing") is None

    def test_duplicate_name_rejected(self):
        eng = RuleEngine()
        eng.add_rule(make_rule("a", lambda c: True))
        with pytest.raises(ValueError):
            eng.add_rule(make_rule("a", lambda c: True))

    def test_remove(self, engine):
        assert engine.remove_rule("wall_ok")
        assert not engine.remove_rule("wall_ok")

    def test_evaluate_triggers(self, engine):
        results = engine.evaluate({"wall_thickness": 0.5})
        by_name = {r.name: r for r in results}
        assert by_name["wall_too_thin"].triggered
        assert not by_name["wall_ok"].triggered

    def test_evaluate_single(self, engine):
        result = engine.evaluate_single("wall_too_thin", {"wall_thickness": 0.5})
        assert isinstance(result, RuleResult)
        assert result.triggered

    def test_select_unknown_raises(self, engine):
        with pytest.raises(KeyError):
            engine.evaluate({}, rule_names=["nope"])

    def test_violations(self, engine):
        violations = engine.violations({"wall_thickness": 0.5})
        assert [v.name for v in violations] == ["wall_too_thin"]

    def test_summary(self, engine):
        summary = engine.summary({"wall_thickness": 0.5})
        assert summary["triggered"] == 1
        assert summary["by_severity"]["error"] == 1


class TestForwardChaining:
    def test_actions_mutate_context(self):
        eng = RuleEngine()

        def seed_action(ctx):
            ctx["derived"] = ctx.get("base", 0) * 2
            return None

        def check_derived(ctx):
            return ctx.get("derived", -1) >= 10

        eng.add_rule(Rule(name="seed", condition=lambda c: True, action=seed_action))
        eng.add_rule(make_rule("derived_ok", check_derived, severity="info"))

        results = eng.run({"base": 6})
        assert any(r.name == "derived_ok" and r.triggered for r in results)

    def test_no_infinite_loop(self):
        eng = RuleEngine()
        eng.add_rule(Rule(name="once", condition=lambda c: True, action=lambda c: None))
        results = eng.run({}, max_rounds=100)
        assert len(results) == 1

    def test_max_rounds_validation(self):
        eng = RuleEngine()
        with pytest.raises(ValueError):
            eng.run({}, max_rounds=0)
