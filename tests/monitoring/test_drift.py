"""tests/monitoring/test_drift.py"""

from __future__ import annotations

import pytest

from cadgenesis.monitoring.drift import (
    DriftMetric,
    FeatureDriftMonitor,
    compute_drift,
)


def test_identical_distributions_zero():
    ref = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert compute_drift(ref, ref, bins=10) == pytest.approx(0.0, abs=1e-6)


def test_shifted_distribution_positive():
    ref = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    cur = [9.0, 9.0, 9.0, 10.0, 10.0, 10.0]
    assert compute_drift(ref, cur, bins=10) > 0


def test_empty_current_zero():
    assert compute_drift([1.0, 2.0], [], bins=5) == 0.0


def test_kl_and_js_metrics():
    ref = [1.0, 1.0, 2.0, 2.0]
    cur = [1.2, 0.8, 2.1, 1.9]
    for metric in (DriftMetric.KL, DriftMetric.JS):
        score = compute_drift(ref, cur, bins=5, metric=metric)
        assert score >= 0


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        compute_drift([1], [1], metric="bogus")


def test_feature_monitor_update_and_evaluate():
    monitor = FeatureDriftMonitor(
        reference={"length": [1.0, 2.0, 3.0, 4.0]},
        threshold=0.2,
        bins=5,
    )
    monitor.update("length", [1.0, 2.0, 3.0, 4.0])
    report = monitor.evaluate()["length"]
    assert not report.drifted
    assert report.samples_reference == 4
    assert report.samples_current == 4


def test_feature_monitor_detects_drift():
    monitor = FeatureDriftMonitor(
        reference={"length": [1.0, 1.0, 1.0, 1.0]},
        threshold=0.05,
        bins=4,
        limits={"length": (0.0, 10.0)},
    )
    monitor.update("length", [9.0, 9.0, 9.0, 9.0])
    assert monitor.any_drifted()
    assert monitor.evaluate()["length"].drifted


def test_feature_monitor_unknown_feature():
    monitor = FeatureDriftMonitor(reference={"a": [1.0]})
    with pytest.raises(KeyError):
        monitor.update("b", [1.0])
    with pytest.raises(KeyError):
        monitor.score("b")


def test_feature_monitor_reset():
    monitor = FeatureDriftMonitor(reference={"a": [1.0, 2.0, 3.0]})
    monitor.update("a", [9.0, 9.0])
    monitor.reset("a")
    assert monitor.evaluate()["a"].samples_current == 0


def test_feature_monitor_invalid_threshold():
    with pytest.raises(ValueError):
        FeatureDriftMonitor(reference={"a": [1.0]}, threshold=-1)


def test_drift_report_to_dict():
    monitor = FeatureDriftMonitor(reference={"a": [1.0, 2.0]})
    monitor.update("a", [1.5, 1.5])
    report = monitor.evaluate()["a"]
    d = report.to_dict()
    assert d["feature"] == "a"
    assert d["metric"] == "psi"
    assert "score" in d and "drifted" in d
