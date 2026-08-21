"""
cadgenesis.platform.monitoring
==============================
Production monitoring integration for the CADGenesis-LM platform.

- Prometheus: pure-Python text-format exporter rendered from the existing
  ``telemetry.metrics.MetricsRegistry`` (no external dependency), served from
  the REST API at ``/metrics``
- OpenTelemetry: optional OTLP bridge (``opentelemetry-sdk``) used only when
  installed, with a no-op fallback
- Grafana: dashboard provisioning JSON (``grafana_dashboard()``)
- Health aggregation: builds a full health payload from ``HealthChecker``
"""

from __future__ import annotations

import logging
from typing import Any

from cadgenesis.monitoring.health import HealthChecker
from cadgenesis.telemetry.metrics import MetricsRegistry

logger = logging.getLogger("cadgenesis.platform.monitoring")


def _format_name(name: str) -> str:
    """Prometheus metric name: alpha/underscore, dot->underscore, prefix."""
    cleaned = name.replace(".", "_").replace("-", "_")
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}"
    return cleaned


def render_prometheus(registry: MetricsRegistry, prefix: str = "cadgenesis") -> str:
    """Render all registry metrics in Prometheus text exposition format."""
    lines: list[str] = []
    snapshot = registry.snapshot()
    for metric in snapshot.get("metrics", []):
        name = metric.get("name", "")
        formatted = _format_name(name)
        if not formatted.startswith(prefix):
            formatted = f"{prefix}_{formatted}"
        labels = metric.get("labels") or {}
        label_suffix = "".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        if label_suffix:
            label_suffix = "{" + label_suffix + "}"
        metric_type = metric.get("type")
        lines.append(f"# TYPE {formatted} {metric_type}")
        if metric_type == "counter" or metric_type == "gauge":
            lines.append(f"{formatted}{label_suffix} {metric.get('value', 0)}")
        elif metric_type == "histogram":
            total = 0
            for bucket in metric.get("buckets", []):
                bound = bucket.get("le")
                count = int(bucket.get("count", 0))
                total += count
                le = "+Inf" if bound == float("inf") else bound
                lines.append(f'{formatted}_bucket{{le="{le}"}} {total}')
            lines.append(f"{formatted}_sum {metric.get('sum', 0)}")
            lines.append(f"{formatted}_count {metric.get('count', 0)}")
    return "\n".join(lines) + "\n" if lines else ""


class PrometheusExporter:
    """Exports a :class:`MetricsRegistry` snapshot as Prometheus text."""

    def __init__(self, registry: MetricsRegistry, prefix: str = "cadgenesis") -> None:
        self.registry = registry
        self.prefix = prefix

    def render(self) -> str:
        return render_prometheus(self.registry, prefix=self.prefix)


class HealthAggregator:
    """Wraps :class:`HealthChecker` and renders status payloads for APIs."""

    def __init__(self, checker: HealthChecker) -> None:
        self.checker = checker

    def summary(self) -> dict[str, Any]:
        return self.checker.summary()

    def full_report(self) -> dict[str, Any]:
        results = [
            {
                "name": result.name,
                "ok": result.ok,
                "status": result.status.value
                if hasattr(result.status, "value")
                else str(result.status),
                "detail": result.detail,
                "data": result.data,
            }
            for result in self.checker.run()
        ]
        return {"status": self.checker.summary()["status"], "checks": results}


class OpenTelemetryBridge:
    """Optional OTLP exporter bridge; no-op when OpenTelemetry is absent."""

    def __init__(self, endpoint: str | None = None, service_name: str = "cadgenesis") -> None:
        self.endpoint = endpoint
        self.service_name = service_name
        self._enabled = False
        self._exporter: Any = None
        self._provider: Any = None
        if endpoint:
            self._try_initialize()

    def _try_initialize(self) -> None:
        try:  # pragma: no cover - optional dependency
            from opentelemetry import trace  # type: ignore[import-not-found]
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,  # type: ignore[import-not-found]
            )

            resource = Resource.create({"service.name": self.service_name})
            self._provider = TracerProvider(resource=resource)
            self._exporter = OTLPSpanExporter(endpoint=self.endpoint)
            self._provider.add_span_processor(BatchSpanProcessor(self._exporter))
            trace.set_tracer_provider(self._provider)
            self._enabled = True
            logger.info("OpenTelemetry exporter enabled -> %s", self.endpoint)
        except ImportError:
            logger.warning(
                "OpenTelemetry export requested but not installed; "
                "install 'opentelemetry-sdk' + 'opentelemetry-exporter-otlp'"
            )

    def enabled(self) -> bool:
        return self._enabled


def grafana_dashboard(title: str = "CADGenesis-LM", uid: str = "cadgenesis") -> dict[str, Any]:
    """A Grafana dashboard provisioning JSON skeleton for platform metrics."""
    panels = [
        {
            "id": 1,
            "type": "timeseries",
            "title": "Inference requests",
            "targets": [
                {
                    "expr": "rate(cadgenesis_inference_requests_total[5m])",
                    "legendFormat": "requests",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        },
        {
            "id": 2,
            "type": "timeseries",
            "title": "Inference latency (p95)",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, sum(rate("
                    "cadgenesis_inference_latency_seconds_bucket[5m])) by (le))"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        },
        {
            "id": 3,
            "type": "timeseries",
            "title": "Request failures",
            "targets": [{"expr": "rate(cadgenesis_inference_errors_total[5m])"}],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        },
    ]
    return {
        "title": title,
        "uid": uid,
        "panels": panels,
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
    }


__all__ = [
    "HealthAggregator",
    "OpenTelemetryBridge",
    "PrometheusExporter",
    "grafana_dashboard",
    "render_prometheus",
]
