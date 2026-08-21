"""cadgenesis.telemetry.logs
=========================
Structured telemetry log emission and an event log buffer for CADGenesis-LM
v6.0.  Builds on :mod:`cadgenesis.logging` for delivery while providing
in-process capture for dashboards and tests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.logging.emitter import StructuredLogEmitter

_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


@dataclass
class TelemetryEvent:
    """A captured telemetry event."""

    level: str
    message: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "message": self.message, **self.fields}


class EventBuffer:
    """Thread-safe bounded buffer of recent telemetry events."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._events: list[TelemetryEvent] = []
        self._lock = threading.Lock()

    def append(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.capacity:
                del self._events[: len(self._events) - self.capacity]

    def drain(self) -> list[TelemetryEvent]:
        with self._lock:
            events, self._events = self._events, []
            return events

    def snapshot(self) -> list[TelemetryEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class TelemetryLogger:
    """Emits structured telemetry logs and optionally buffers them.

    Usage::

        telemetry = TelemetryLogger("inference")
        telemetry.log("info", "generated", tokens=32, latency_ms=4.2)
        telemetry.buffer.drain()   # access captured events
    """

    def __init__(
        self,
        name: str,
        capture: bool = False,
        buffer_capacity: int = 1000,
        default_fields: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.capture = capture
        self.buffer = EventBuffer(buffer_capacity)
        self._emitter = StructuredLogEmitter(name, default_fields or {})

    def log(self, level: str, message: str, **fields: Any) -> None:
        """Emit a telemetry event and capture it when ``capture`` is enabled."""
        level = level.lower()
        if level not in _LOG_LEVELS:
            raise ValueError(f"invalid level {level!r}; expected one of {_LOG_LEVELS}")
        if self.capture:
            self.buffer.append(TelemetryEvent(level, message, fields))
        self._emitter.emit(level, message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self.log("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self.log("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self.log("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.log("error", message, **fields)

    def critical(self, message: str, **fields: Any) -> None:
        self.log("critical", message, **fields)


def log_event(level: str, message: str, **fields: Any) -> None:
    """Module-level convenience for emitting a telemetry log line."""
    StructuredLogEmitter("telemetry.events").emit(level, message, **fields)
