from __future__ import annotations

from cadgenesis.monitoring.health import HealthChecker, HealthResult
from cadgenesis.platform.monitoring import (
    HealthAggregator,
    PrometheusExporter,
    grafana_dashboard,
    render_prometheus,
)
from cadgenesis.telemetry.metrics import MetricsRegistry


class TestPrometheusExporter:
    def test_render_counts_and_gauges(self):
        registry = MetricsRegistry()
        counter = registry.counter("requests", "inference requests")
        counter.inc()
        counter.inc()
        gauge = registry.gauge("active", "active models")
        gauge.set(2)
        text = PrometheusExporter(registry).render()
        assert "# TYPE cadgenesis_requests counter" in text
        assert "cadgenesis_requests 2.0" in text
        assert "# TYPE cadgenesis_active gauge" in text
        assert "cadgenesis_active 2.0" in text

    def test_render_histogram(self):
        registry = MetricsRegistry()
        hist = registry.histogram("latency", "ms", buckets=[0.1, 1.0, 10.0, float("inf")])
        hist.observe(0.2)
        hist.observe(5.0)
        text = PrometheusExporter(registry).render()
        assert "cadgenesis_latency_bucket" in text
        assert 'le="0.1"' in text
        assert "cadgenesis_latency_count 2" in text

    def test_render_uses_snapshot(self):
        registry = MetricsRegistry()
        registry.counter("requests", "inference requests").inc()
        text = render_prometheus(registry)
        assert "cadgenesis_requests 1.0" in text


class TestHealthAggregator:
    def test_summary_healthy(self):
        checker = HealthChecker()
        checker.register("disk", lambda: HealthResult("disk", True, "ok"))
        aggregator = HealthAggregator(checker)
        summary = aggregator.summary()
        assert summary["status"] == "healthy"
        assert summary["checks"][0]["name"] == "disk"

    def test_summary_degraded(self):
        checker = HealthChecker()
        checker.register("ok", lambda: HealthResult("ok", True, "fine"))
        checker.register("bad", lambda: HealthResult("bad", False, "broken"))
        aggregator = HealthAggregator(checker)
        assert aggregator.summary()["status"] == "degraded"

    def test_summary_unhealthy(self):
        checker = HealthChecker()
        checker.register("bad", lambda: HealthResult("bad", False, "broken"))
        aggregator = HealthAggregator(checker)
        assert aggregator.summary()["status"] == "unhealthy"

    def test_full_report(self):
        checker = HealthChecker()
        aggregator = HealthAggregator(checker)
        report = aggregator.full_report()
        assert "status" in report and "checks" in report


class TestGrafanaDashboard:
    def test_structure(self):
        dashboard = grafana_dashboard("CADGenesis", uid="cadgenesis")
        assert dashboard["title"] == "CADGenesis"
        assert dashboard["uid"] == "cadgenesis"
        assert "panels" in dashboard
        assert dashboard["panels"]
