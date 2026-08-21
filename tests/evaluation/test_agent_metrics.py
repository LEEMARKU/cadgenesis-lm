"""tests/evaluation/test_agent_metrics.py
========================================
Unit tests for the Pillar 5 agent metrics.
"""

from __future__ import annotations

import pytest

from cadgenesis.evaluation import (
    consensus_agreement,
    error_rate,
    fleet_coverage,
    mean_latency,
    p95_latency,
    pipeline_success,
    run_agent_benchmark,
    task_success_rate,
)


def test_task_success_rate():
    results = [{"ok": True}, {"ok": True}, {"ok": False}]
    assert task_success_rate(results) == pytest.approx(2 / 3)
    assert task_success_rate([]) == 0.0


def test_error_rate():
    assert error_rate([{"ok": True}, {"ok": False}]) == pytest.approx(0.5)


def test_latency_metrics():
    assert mean_latency([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert mean_latency([]) == 0.0
    assert p95_latency([1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.0)


def test_consensus_agreement():
    summaries = [
        {"decision": "yes", "majority": "yes"},
        {"decision": "no", "majority": "yes"},
    ]
    assert consensus_agreement(summaries) == pytest.approx(0.5)


def test_fleet_coverage():
    snapshot = [
        {"role": "a", "capabilities": [1, 2]},
        {"role": "b", "capabilities": [1]},
    ]
    coverage = fleet_coverage(snapshot)
    assert coverage["role_count"] == 2
    assert coverage["capability_count"] == 3


def test_pipeline_success():
    reports = [
        {"validation": {"passed": True}},
        {"validation": {"passed": False}},
    ]
    assert pipeline_success(reports) == pytest.approx(0.5)


def test_run_agent_benchmark():
    report = run_agent_benchmark(
        results=[{"ok": True}, {"ok": True}],
        durations=[0.1, 0.2],
        summaries=[{"decision": "y", "majority": "y"}],
        registry_snapshot=[{"role": "a", "capabilities": []}],
        pipeline_reports=[{"validation": {"passed": True}}],
    )
    assert report["task_success_rate"] == 1.0
    assert report["checks"]["results"] == 2
