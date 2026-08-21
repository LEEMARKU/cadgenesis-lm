"""
cadgenesis.platform.logging
===========================
Distributed logging & log aggregation for the CADGenesis-LM platform.

- Structured log emission on top of the existing ``logging`` subsystem
  (``cadgenesis.logging.StructuredLogEmitter``)
- ``LogAggregator``: an in-process collector that fans structured records
  out to sinks (stdout JSON, file, HTTP(S) endpoint, callback), thread-safe
  and bounded
- ``DistributedLogClient``: ships JSON log records to a remote aggregator
  over HTTP(S) using only the standard library
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from cadgenesis.logging.emitter import StructuredLogEmitter

logger = logging.getLogger("cadgenesis.platform.logging")

LogSink = Callable[[dict[str, Any]], None]


class LogAggregator:
    """Bounded, thread-safe collector distributing structured records to sinks."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = max(100, capacity)
        self._sinks: list[LogSink] = []
        self._buffer: deque[dict[str, Any]] = deque(maxlen=self.capacity)
        self._lock = threading.RLock()

    def add_sink(self, sink: LogSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: LogSink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def emit(self, record: Mapping[str, Any]) -> None:
        entry = dict(record)
        entry.setdefault("ts", time.time())
        with self._lock:
            self._buffer.append(entry)
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(entry)
            except Exception:
                logger.exception("log sink failed")

    def flush(self) -> None:
        """Drain buffered records to all sinks (sinks must be idempotent)."""
        with self._lock:
            pending = list(self._buffer)
        for entry in pending:
            for sink in list(self._sinks):
                try:
                    sink(entry)
                except Exception:
                    logger.exception("log sink failed during flush")

    def snapshot(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            data = list(self._buffer)
        return data[-limit:] if limit else data

    def __len__(self) -> int:
        return len(self._buffer)


class JsonFileSink:
    """Append JSON lines to a file (one record per line)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class HttpSink:
    """Ship records to a remote aggregator endpoint (POST JSON, stdlib only)."""

    def __init__(
        self, url: str, timeout: float = 5.0, headers: Mapping[str, str] | None = None
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def __call__(self, record: Mapping[str, Any]) -> None:
        body = json.dumps(record, default=str).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status >= 300:
                raise OSError(f"aggregator returned {response.status}")


class DistributedLogClient:
    """Small HTTP client to ship structured records to a remote aggregator."""

    def __init__(self, endpoint: str, service: str = "cadgenesis", timeout: float = 5.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.service = service
        self.timeout = timeout

    def send(self, record: Mapping[str, Any]) -> None:
        payload = {"service": self.service, **dict(record)}
        HttpSink(f"{self.endpoint}/logs", timeout=self.timeout)(payload)

    def send_batch(self, records: Iterable[Mapping[str, Any]]) -> int:
        batch = [{"service": self.service, **dict(r)} for r in records]
        body = json.dumps(batch, default=str).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/logs/batch",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status
        except urllib.error.URLError as exc:
            logger.warning("log batch delivery failed: %s", exc)
            return -1


def structured_record(
    level: str,
    message: str,
    service: str = "cadgenesis",
    fields: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a structured JSON log record dict."""
    return {
        "ts": time.time(),
        "level": level.upper(),
        "service": service,
        "message": message,
        **(fields or {}),
        **extra,
    }


def wire_aggregator(aggregator: LogAggregator) -> StructuredLogEmitter:
    """Attach an aggregator to the project structured emitter as a sink."""

    class _AggregatorSink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            excluded = {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "taskName",
                "message",
            }
            fields = {key: value for key, value in record.__dict__.items() if key not in excluded}
            aggregator.emit(
                structured_record(record.levelname.lower(), record.getMessage(), fields=fields)
            )

    emitter = StructuredLogEmitter(name="cadgenesis.platform.aggregator")
    emitter._logger.addHandler(_AggregatorSink())
    return emitter


__all__ = [
    "DistributedLogClient",
    "HttpSink",
    "JsonFileSink",
    "LogAggregator",
    "structured_record",
    "wire_aggregator",
]
