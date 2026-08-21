from __future__ import annotations

import pytest

from cadgenesis.research.ablation import (
    ABLATION_KINDS,
    AblationEngine,
    AblationResult,
    AblationSpec,
)


def scorer(config):
    """Runner: loss degrades when a component is ablated."""
    disabled = sum(
        1 for key, value in config.items() if key.startswith("ablate.") and value is True
    )
    layers = len(config.get("ablate.layers", []))
    return {"loss": 0.5 + 0.1 * disabled + 0.05 * layers, "acc": 0.9 - 0.05 * disabled}


class TestAblationSpec:
    def test_to_dict(self):
        spec = AblationSpec(kind="component", target="lora")
        data = spec.to_dict()
        assert data["kind"] == "component"
        assert data["target"] == "lora"


class TestAblationEngine:
    def test_run_computes_deltas(self):
        engine = AblationEngine(runner=scorer, metric_keys=["loss"], baseline_config={})
        results = engine.run([AblationSpec(kind="component", target="lora")])
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, AblationResult)
        assert result.deltas["loss"] == pytest.approx(0.1)

    def test_config_mutation_by_kind(self):
        AblationEngine(runner=scorer, metric_keys=["loss"])
        cases = {
            "component": {"ablate.lora": True},
            "layer": {"ablate.layers": ["0"]},
            "attention": {"model.heads": {"geometry": 0}},
            "memory": {"memory.disabled": "episodic"},
            "agent": {"ablate.agents": ["executor"]},
        }
        for kind, expected in cases.items():
            mutated = AblationEngine._apply(
                {},
                AblationSpec(
                    kind=kind,
                    target={
                        "layer": "0",
                        "attention": "geometry",
                        "memory": "episodic",
                        "agent": "executor",
                        "component": "lora",
                    }[kind],
                ),
            )
            assert mutated == expected, kind

    def test_unknown_kind(self):
        engine = AblationEngine(runner=scorer)
        with pytest.raises(ValueError):
            engine._apply({}, AblationSpec(kind="nope", target="x"))

    def test_summary_most_impactful(self):
        engine = AblationEngine(runner=scorer, metric_keys=["loss"])
        results = engine.run(
            [
                AblationSpec(kind="component", target="a"),
                AblationSpec(kind="layer", target="0"),
                AblationSpec(kind="layer", target="1"),
            ]
        )
        summary = engine.summary(results)
        assert summary["count"] == 3
        assert summary["most_impactful"]["target"] in {"a", "0", "1"}

    def test_empty_summary(self):
        engine = AblationEngine(runner=scorer)
        summary = engine.summary([])
        assert summary["count"] == 0
        assert summary["most_impactful"] is None


class TestKinds:
    def test_kinds(self):
        assert set(ABLATION_KINDS) == {"component", "layer", "attention", "memory", "agent"}
