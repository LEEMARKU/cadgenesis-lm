"""
tests/evaluation/test_repair_metrics.py
=======================================
Tests for the completed self-correction metrics: initial success rate,
repair success rate, iterations to success, mean attempts
(pre-training gate: self-correction system).
"""

from __future__ import annotations

import pytest

from cadgenesis.evaluation.execution_metrics import (
    initial_success_rate,
    iterations_to_success,
    mean_attempts,
    repair_rate,
    repair_success_rate,
    run_execution_benchmark,
)


def _result(
    valid: bool = True,
    repaired: bool = False,
    iterations: int | None = None,
    errors: list[str] | None = None,
):
    report = {"attempted": repaired, "applied": repaired}
    if iterations is not None:
        report["iterations"] = iterations
    return type(
        "R",
        (),
        {
            "is_valid_geometry": valid,
            "repair_report": report,
            "errors": errors or [],
            "estimated_cost_usd": 10.0,
        },
    )()


class TestInitialSuccessRate:
    def test_all_clean_first_try(self):
        results = [_result(), _result()]
        assert initial_success_rate(results) == 1.0

    def test_repaired_not_initial(self):
        results = [_result(repaired=True), _result()]
        assert initial_success_rate(results) == 0.5

    def test_errors_not_initial(self):
        results = [_result(errors=["boom"]), _result()]
        assert initial_success_rate(results) == 0.5

    def test_invalid_not_initial(self):
        results = [_result(valid=False), _result()]
        assert initial_success_rate(results) == 0.5

    def test_empty(self):
        assert initial_success_rate([]) == 0.0


class TestRepairSuccessRate:
    def test_all_repairs_succeed(self):
        results = [_result(repaired=True), _result(repaired=True)]
        assert repair_success_rate(results) == 1.0

    def test_half_repairs_succeed(self):
        results = [_result(repaired=True), _result(repaired=True, valid=False)]
        assert repair_success_rate(results) == 0.5

    def test_no_repairs_zero(self):
        assert repair_success_rate([_result(), _result()]) == 0.0

    def test_empty(self):
        assert repair_success_rate([]) == 0.0


class TestIterationsToSuccess:
    def test_first_try_is_one(self):
        assert iterations_to_success([_result(), _result()]) == 1.0

    def test_mean_over_succeeded(self):
        results = [
            _result(iterations=3),  # valid after 3 iterations
            _result(iterations=1),  # valid first try
            _result(valid=False, iterations=5),  # never valid -> excluded
        ]
        assert iterations_to_success(results) == pytest.approx(2.0)

    def test_none_succeeded(self):
        assert iterations_to_success([_result(valid=False)]) == 0.0

    def test_empty(self):
        assert iterations_to_success([]) == 0.0

    def test_repair_report_iterations(self):
        result = _result(repaired=True, iterations=2)
        assert iterations_to_success([result]) == pytest.approx(2.0)


class TestMeanAttempts:
    def test_all_results_included(self):
        results = [
            _result(iterations=1),
            _result(iterations=4, valid=False),
        ]
        assert mean_attempts(results) == pytest.approx(2.5)

    def test_empty(self):
        assert mean_attempts([]) == 0.0


class TestRunExecutionBenchmark:
    def test_repair_metrics_in_report(self):
        results = [
            _result(repaired=True, iterations=2),
            _result(valid=False, repaired=True, iterations=4),
            _result(),
        ]
        report = run_execution_benchmark(results)
        assert report["initial_success_rate"] == pytest.approx(1 / 3, abs=0.001)
        assert report["repair_success_rate"] == pytest.approx(0.5, abs=0.001)
        assert report["iterations_to_success"] == pytest.approx(1.5, abs=0.001)
        assert report["mean_attempts"] == pytest.approx(7 / 3, abs=0.001)
        assert report["repair_rate"] == pytest.approx(2 / 3, abs=0.001)