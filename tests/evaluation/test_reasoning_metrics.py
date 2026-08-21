"""Tests for Pillar 7 reasoning evaluation metrics."""

from __future__ import annotations

import pytest

from cadgenesis.evaluation import (
    constraint_reasoning,
    engineering_correctness,
    manufacturing_correctness,
    reasoning_accuracy,
    rule_utilization,
    run_reasoning_benchmark,
    symbolic_consistency,
    topology_reasoning,
)


def test_reasoning_accuracy() -> None:
    assert reasoning_accuracy([True, True, False], [True, True, False]) == 1.0
    assert reasoning_accuracy([True, False], [True, True]) == 0.5
    assert reasoning_accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        reasoning_accuracy([True], [True, False])


def test_symbolic_consistency() -> None:
    assert symbolic_consistency(3, 4) == 0.75
    assert symbolic_consistency(0, 0) == 0.0


def test_rule_utilization() -> None:
    assert rule_utilization(2, 10) == 0.2
    assert rule_utilization(0, 5) == 0.0
    assert rule_utilization(1, 0) == 0.0


def test_engineering_correctness() -> None:
    assert engineering_correctness([("ISO 286", True), ("ASME Y14.5", False)]) == 0.5
    assert engineering_correctness([]) == 0.0


def test_manufacturing_correctness() -> None:
    assert manufacturing_correctness([("wall", True), ("hole", True)]) == 1.0
    assert manufacturing_correctness([("wall", False)]) == 0.0


def test_constraint_reasoning() -> None:
    assert constraint_reasoning([(True, 0.0), (False, 5.0)]) == 0.5
    assert constraint_reasoning([]) == 0.0


def test_topology_reasoning() -> None:
    assert topology_reasoning([(True, 4), (True, 6)]) == 1.0
    assert topology_reasoning([(True, 4), (False, 4)]) == 0.5


def test_run_reasoning_benchmark() -> None:
    report = run_reasoning_benchmark(
        predictions=[(True, True), (False, False)],
        conclusions=(2, 2),
        rule_usage=(1, 4),
        compliance=[("ISO", True)],
        dfm_checks=[("wall", True), ("draft", False)],
        constraint_batches=[(True, 0.0)],
        topology_samples=[(True, 4)],
    )
    assert report["reasoning_accuracy"] == 1.0
    assert report["symbolic_consistency"] == 1.0
    assert report["rule_utilization"] == 0.25
    assert report["engineering_correctness"] == 1.0
    assert report["manufacturing_correctness"] == 0.5
    assert report["constraint_reasoning"] == 1.0
    assert report["topology_reasoning"] == 1.0
    assert report["checks"]["predictions"] == 2


def test_run_reasoning_benchmark_empty() -> None:
    report = run_reasoning_benchmark()
    assert report["reasoning_accuracy"] == 0.0
    assert report["checks"]["predictions"] == 0
