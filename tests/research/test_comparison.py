from __future__ import annotations

import itertools

import pytest

from cadgenesis.research.comparison import ComparisonReport, ModelComparator, VariantResult

_jitter = itertools.count()


def runner(variant, config):
    jitter = next(_jitter) * 0.001  # deterministic per-repeat variation
    if variant == "baseline":
        return {"loss": 0.5 + jitter, "acc": 0.9}
    if variant == "ablated":
        return {"loss": 0.8 + jitter, "acc": 0.7}
    return {"loss": 0.6 + jitter, "acc": 0.85}


class TestComparisonReport:
    def test_add_and_rankings(self):
        report = ComparisonReport(dimension="checkpoint")
        report.add(VariantResult(variant="a", dimension="checkpoint", metrics={"loss": 0.4}))
        report.add(VariantResult(variant="b", dimension="checkpoint", metrics={"loss": 0.2}))
        rankings = report.rankings("loss", minimize=True)
        assert rankings == [("b", 0.2), ("a", 0.4)]

    def test_to_dict_and_json(self):
        report = ComparisonReport(dimension="adapter")
        report.add(VariantResult(variant="a", dimension="adapter", metrics={"loss": 0.4}))
        data = report.to_dict()
        assert data["dimension"] == "adapter"
        assert data["variants"][0]["variant"] == "a"
        assert '"dimension": "adapter"' in report.to_json()


class TestVariantResult:
    def test_intervals_require_repeats(self):
        result = VariantResult(
            variant="a",
            dimension="x",
            metrics={"loss": 0.5},
            repeats=[{"loss": 0.5}, {"loss": 0.6}],
        )
        intervals = result.compute_intervals()
        assert "loss" in intervals
        assert intervals["loss"]["lower"] < intervals["loss"]["upper"]

    def test_intervals_single_repeat(self):
        result = VariantResult(
            variant="a", dimension="x", metrics={"loss": 0.5}, repeats=[{"loss": 0.5}]
        )
        assert result.compute_intervals() == {}


class TestModelComparator:
    def test_compare_with_repeats(self):
        comparator = ModelComparator(runner=runner, repeats=3, seed=1)
        report = comparator.compare(
            "checkpoint", ["baseline", "ablated"], metric="loss", minimize=True
        )
        assert len(report.variants) == 2
        assert report.variants[0].metrics["loss"] == pytest.approx(0.5, abs=0.01)
        ci = report.variants[0].confidence_intervals["loss"]
        assert ci["lower"] <= 0.5 <= ci["upper"]

    def test_significance_matrix(self):
        comparator = ModelComparator(runner=runner, repeats=3, seed=1)
        report = comparator.compare(
            "checkpoint", ["baseline", "ablated"], metric="loss", minimize=True, alpha=0.05
        )
        entry = report.significance["baseline"]["ablated"]
        assert entry["significant"] is True
        assert entry["p_value"] < 0.05

    def test_unknown_dimension(self):
        comparator = ModelComparator(runner=runner)
        with pytest.raises(ValueError):
            comparator.compare("bogus", ["baseline"])
