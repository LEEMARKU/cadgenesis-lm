"""cadgenesis.telemetry.tracing
============================
Distributed tracing for CADGenesis-LM v6.0: hierarchical spans, context
propagation, and a lightweight OpenTelemetry-compatible surface.

No external tracing SDK is required; ``Tracer``/``Span`` provide the core
semantics and can be exported to any backend.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import time
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_TRACING_ENABLED = True
_tracing_lock = threading.Lock()


def set_tracing_enabled(enabled: bool) -> None:
    """Globally enable or disable span recording."""
    global _TRACING_ENABLED
    with _tracing_lock:
        _TRACING_ENABLED = enabled


def tracing_enabled() -> bool:
    return _TRACING_ENABLED


@dataclass
class SpanContext:
    """Identity and trace linkage for a span."""

    trace_id: str
    span_id: str
    parent_id: str | None = None


@dataclass
class Span:
    """A recorded span with timing, attributes and a (nested) child list."""

    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if not self.context.span_id:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return (end - self.start_time) * 1000.0

    def finish(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_id": self.context.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 4),
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children],
        }


class Tracer:
    """Creates and manages spans, tracking a per-thread active span stack."""

    def __init__(self, name: str = "cadgenesis") -> None:
        self.name = name
        self._local = threading.local()
        self._completed: list[Span] = []
        self._lock = threading.Lock()

    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Start a new span, nested under the current active span (if any)."""
        if not tracing_enabled():
            return Span(name, SpanContext("", ""))
        active = self.current_span()
        parent_id = active.context.span_id if active else None
        trace_id = active.context.trace_id if active else uuid.uuid4().hex
        span_id = uuid.uuid4().hex
        span = Span(name=name, context=SpanContext(trace_id, span_id, parent_id))
        if attributes:
            span.attributes.update(attributes)
        if active is not None:
            active.children.append(span)
        self._stack().append(span)
        return span

    def end_span(self, span: Span) -> Span:
        """Finish a span and pop it off the active stack if it is on top."""
        span.finish()
        stack = self._stack()
        if stack and stack[-1] is span:
            stack.pop()
            if span.context.parent_id is None:
                with self._lock:
                    self._completed.append(span)
        elif span.context.parent_id is None:
            with self._lock:
                if span not in self._completed:
                    self._completed.append(span)
        return span

    def current_span(self) -> Span | None:
        stack = self._stack()
        return stack[-1] if stack else None

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager creating and finishing a span.

        Usage::

            with tracer.span("forward", {"layer": 3}):
                model(...)
        """
        span = self.start_span(name, attributes)
        try:
            yield span
        except BaseException:
            span.finish()
            raise
        finally:
            self.end_span(span)

    def trace(
        self,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Callable[[F], F]:
        """Decorator wrapping a function in a span named after it."""

        def decorator(func: F) -> F:
            span_name = name or str(getattr(func, "__qualname__", func.__name__))

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, attributes):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def finish_all(self) -> list[Span]:
        """Finish and return all open and completed root spans in this thread.

        Children remain reachable through their root's ``children`` list.
        """
        stack = self._stack()
        open_roots = [s for s in stack if s.context.parent_id is None]
        for span in list(stack):
            span.finish()
        stack.clear()
        with self._lock:
            completed = list(self._completed)
            self._completed = []
        return open_roots + completed
