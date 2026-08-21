"""tests/telemetry/test_tracing.py"""

from __future__ import annotations

from cadgenesis.telemetry.tracing import (
    Tracer,
    set_tracing_enabled,
    tracing_enabled,
)


def _reset():
    set_tracing_enabled(True)


def test_tracing_enabled_toggle():
    _reset()
    assert tracing_enabled()
    set_tracing_enabled(False)
    assert not tracing_enabled()
    _reset()


def test_span_context_manager():
    tracer = Tracer("test")
    with tracer.span("op") as span:
        span.set_attribute("layer", 1)
    assert span.end_time is not None
    assert span.duration_ms >= 0
    assert span.name == "op"
    assert span.attributes["layer"] == 1


def test_nested_spans():
    tracer = Tracer("test")
    with tracer.span("outer") as outer, tracer.span("inner") as inner:
        pass
    assert inner.context.parent_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id
    assert len(outer.children) == 1
    assert outer.children[0].name == "inner"


def test_trace_decorator():
    tracer = Tracer("test")

    @tracer.trace()
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_span_disabled_records_nothing():
    tracer = Tracer("test")
    set_tracing_enabled(False)
    with tracer.span("op") as span:
        span.set_attribute("x", 1)
    assert span.duration_ms == 0
    _reset()


def test_finish_all():
    tracer = Tracer("test")
    with tracer.span("a"), tracer.span("b"):
        pass
    roots = tracer.finish_all()
    assert len(roots) == 1
    assert tracer.current_span() is None


def test_start_span_nested_attrs():
    tracer = Tracer("test")
    span = tracer.start_span("op", {"k": "v"})
    assert span.attributes == {"k": "v"}
    tracer.end_span(span)


def test_span_to_dict():
    tracer = Tracer("test")
    with tracer.span("op") as span:
        pass
    d = span.to_dict()
    assert d["name"] == "op"
    assert "duration_ms" in d
    assert d["children"] == []
