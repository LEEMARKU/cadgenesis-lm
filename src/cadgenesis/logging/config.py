"""cadgenesis.logging.config
=========================
Logging configuration for CADGenesis-LM v6.0: console / rotating-file / JSON
sinks, configurable levels, and a deterministic ``setup_logging`` entrypoint.
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass
class LoggingConfig:
    """Configuration for the root logger.

    All fields are optional; defaults match the values exposed via environment
    variables so deployments can override without code changes.
    """

    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = _DEFAULT_LOG_FORMAT
    json_format: bool = False
    console_enabled: bool = True
    file_enabled: bool = False
    file_path: str = "outputs/logs/cadgenesis.log"
    file_max_bytes: int = 10 * 1024 * 1024  # 10 MiB
    file_backup_count: int = 5
    propagate: bool = True
    root_name: str = "cadgenesis"

    def __post_init__(self) -> None:
        level = self.level.upper()
        if level not in logging._nameToLevel:
            raise ValueError(
                f"invalid log level {self.level!r}; expected one of {sorted(logging._nameToLevel)}"
            )
        self.level = level

    def as_dict(self) -> dict:
        return asdict(self)


def _json_formatter() -> logging.Formatter:
    """Log formatter emitting single-line JSON records."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, str | int | float | None] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                + f".{int(record.msecs * 1000):03d}Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            for key, value in record.__dict__.items():
                if key.startswith("_") or key in payload:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    payload[key] = value
            return json.dumps(payload, ensure_ascii=False)

    return JsonFormatter()


def _text_formatter(log_format: str) -> logging.Formatter:
    return logging.Formatter(log_format)


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Configure the root ``cadgenesis`` logger and return it.

    Calling this function is idempotent: existing ``cadgenesis`` loggers are
    re-bound to the new handlers, and a duplicate call reconfigures handlers
    instead of stacking them.

    Args:
        config: Logging configuration; defaults to ``LoggingConfig()``.

    Returns:
        The configured root logger.
    """
    cfg = config or LoggingConfig()
    logger = logging.getLogger(cfg.root_name)
    logger.setLevel(cfg.level)
    logger.propagate = cfg.propagate

    formatter = _json_formatter() if cfg.json_format else _text_formatter(cfg.format)
    handlers: list[logging.Handler] = []

    if cfg.console_enabled:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(cfg.level)
        handlers.append(console)

    if cfg.file_enabled:
        path = Path(cfg.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(path),
            maxBytes=cfg.file_max_bytes,
            backupCount=cfg.file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(cfg.level)
        handlers.append(file_handler)

    logger.handlers = handlers
    logger.disabled = False
    return logger


def get_logger(name: str, config: LoggingConfig | None = None) -> logging.Logger:
    """Return a child logger under the ``cadgenesis`` namespace.

    If the root logger has no handlers yet, ``setup_logging`` is invoked with
    the supplied config (or defaults) so logs are always emitted.
    """
    if not name:
        full_name = "cadgenesis"
    elif name.startswith("cadgenesis.") or name == "cadgenesis":
        full_name = name
    else:
        full_name = f"cadgenesis.{name}"
    root = logging.getLogger("cadgenesis")
    if not root.handlers and not root.disabled:
        setup_logging(config)
    return logging.getLogger(full_name)
