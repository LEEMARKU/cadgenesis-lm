"""tests/telemetry/test_metrics.py"""

from __future__ import annotations

import pytest

from cadgenesis.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    StepTimer,
)


def test_counter():
    counter = Counter("reqs")
    counter.inc()
    counter.inc(2)
    assert counter.snapshot()["value"] == 3
    with pytest.raises(ValueError):
        counter.inc(-1)
    counter.reset()
    assert counter.snapshot()["value"] == 0


def test_gauge():
    gauge = Gauge("mem")
    gauge.set(10)
    gauge.inc(2)
    gauge.dec(1)
    assert gauge.snapshot()["value"] == 11


def test_histogram():
    hist = Histogram("latency", buckets=[0.5, 1.0, float("inf")])
    for v in (0.2, 0.8, 2.0):
        hist.observe(v)
    snap = hist.snapshot()
    assert snap["count"] == 3
    assert snap["sum"] == pytest.approx(3.0)
    cumulative = {b["le"]: b["count"] for b in snap["buckets"]}
    assert cumulative[0.5] == 1
    assert cumulative[1.0] == 2
    assert cumulative[float("inf")] == 3
    assert hist.mean == pytest.approx(1.0)


def test_histogram_invalid_buckets():
    with pytest.raises(ValueError):
        Histogram("bad", buckets=[1.0, 0.5, float("inf")])
    with pytest.raises(ValueError):
        Histogram("bad", buckets=[1.0])


def test_histogram_default_buckets():
    hist = Histogram("latency")
    assert hist.bounds[-1] == float("inf")


def test_registry_counter_gauge_histogram():
    registry = MetricsRegistry("test")
    counter = registry.counter("calls")
    gauge = registry.gauge("load")
    hist = registry.histogram("dur")
    counter.inc()
    gauge.set(1.0)
    hist.observe(0.1)
    assert registry.names() == ["test.calls", "test.dur", "test.load"]
    snap = registry.snapshot()
    assert len(snap["metrics"]) == 3
    assert snap["registry"] == "test"


def test_registry_duplicate_rejected():
    registry = MetricsRegistry("test")
    registry.counter("x")
    with pytest.raises(ValueError):
        registry.counter("x")


def test_registry_get():
    registry = MetricsRegistry()
    counter = registry.counter("requests")
    assert registry.get("requests") is counter
    assert registry.get("missing") is None


def test_registry_clear():
    registry = MetricsRegistry()
    registry.counter("a")
    registry.clear()
    assert registry.names() == []


def test_step_timer_records_when_success():
    hist = Histogram("dur")
    with StepTimer(hist):
        pass
    assert hist.snapshot()["count"] == 1
