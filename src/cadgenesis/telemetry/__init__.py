"""cadgenesis.telemetry
====================
Metrics collection, distributed tracing, and structured telemetry logs for
CADGenesis-LM v6.0.
"""

from cadgenesis.telemetry.logs import (
    EventBuffer,
    TelemetryEvent,
    TelemetryLogger,
    log_event,
)
from cadgenesis.telemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricsRegistry,
    MetricType,
    StepTimer,
)
from cadgenesis.telemetry.tracing import (
    Span,
    SpanContext,
    Tracer,
    set_tracing_enabled,
    tracing_enabled,
)

__all__ = [
    "Counter",
    "EventBuffer",
    "Gauge",
    "Histogram",
    "Metric",
    "MetricType",
    "MetricsRegistry",
    "Span",
    "SpanContext",
    "StepTimer",
    "TelemetryEvent",
    "TelemetryLogger",
    "Tracer",
    "log_event",
    "set_tracing_enabled",
    "tracing_enabled",
]
