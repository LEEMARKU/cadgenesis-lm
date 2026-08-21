"""cadgenesis.logging.emitter
==========================
Structured-log emission helpers for CADGenesis-LM v6.0.

``StructuredLogEmitter`` enriches log records with key/value fields so they can
be consumed by JSON pipelines, and ``emit`` provides a thin convenience
function.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.logging.config import LoggingConfig, get_logger

_VALID_LEVELS = ("debug", "info", "warning", "error", "critical")


@dataclass
class StructuredLogEmitter:
    """Thread-safe emitter adding structured fields to log records.

    Usage::

        emitter = StructuredLogEmitter("training", {"pipeline": "main"})
        emitter.emit("info", "step complete", step=10, loss=0.5)
    """

    name: str
    default_fields: Mapping[str, Any] = field(default_factory=dict)
    config: LoggingConfig | None = None
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._logger = self.logger or get_logger(self.name, self.config)

    def emit(self, level: str, message: str, **fields: Any) -> None:
        """Log ``message`` at ``level`` with the given structured fields.

        Args:
            level: One of "debug", "info", "warning", "error", "critical".
            message: Human-readable message.
            **fields: Structured key/value pairs attached to the record.

        Raises:
            ValueError: if ``level`` is not a supported level.
        """
        normalized = level.lower()
        if normalized not in _VALID_LEVELS:
            raise ValueError(f"invalid level {level!r}; expected one of {_VALID_LEVELS}")
        record = self._logger.makeRecord(
            self._logger.name,
            getattr(logging, normalized.upper()),
            __file__,
            0,
            message,
            (),
            None,
            None,
            None,
        )
        with self._lock:
            for key, value in self.default_fields.items():
                setattr(record, key, value)
            for key, value in fields.items():
                setattr(record, key, value)
            self._logger.handle(record)

    def debug(self, message: str, **fields: Any) -> None:
        self.emit("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self.emit("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self.emit("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.emit("error", message, **fields)

    def critical(self, message: str, **fields: Any) -> None:
        self.emit("critical", message, **fields)

    def with_fields(self, **fields: Any) -> StructuredLogEmitter:
        """Return a copy of this emitter with additional default fields."""
        merged = dict(self.default_fields)
        merged.update(fields)
        return StructuredLogEmitter(self.name, merged, self.config, self.logger)


def emit(level: str, message: str, **fields: Any) -> None:
    """Emit a structured log line on the ``cadgenesis.logging.emitter`` logger."""
    StructuredLogEmitter("logging.emitter").emit(level, message, **fields)
