"""tests/logging/test_emitter.py"""

from __future__ import annotations

import io
import logging

from cadgenesis.logging.config import LoggingConfig, setup_logging
from cadgenesis.logging.emitter import StructuredLogEmitter, emit


def _capture(logger_name: str, config: LoggingConfig) -> tuple[StructuredLogEmitter, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("cadgenesis").addHandler(handler)
    emitter = StructuredLogEmitter(logger_name, config=config)
    return emitter, stream


def test_emit_structured_fields():
    setup_logging(LoggingConfig(level="INFO", console_enabled=False))
    emitter = StructuredLogEmitter("test.emitter")
    emitter.info("step done", step=5, loss=0.5)


def test_emit_levels():
    setup_logging(LoggingConfig(level="DEBUG", console_enabled=False))
    emitter = StructuredLogEmitter("test.emitter")
    emitter.debug("d")
    emitter.info("i")
    emitter.warning("w")
    emitter.error("e")
    emitter.critical("c")


def test_emit_invalid_level_raises():
    emitter = StructuredLogEmitter("test.emitter")
    try:
        emitter.emit("bogus", "x")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_with_fields_merges():
    emitter = StructuredLogEmitter("test", default_fields={"project": "A"})
    child = emitter.with_fields(experiment="exp1")
    assert child.default_fields["project"] == "A"
    assert child.default_fields["experiment"] == "exp1"
    assert emitter.default_fields.get("experiment") is None


def test_module_emit_function():
    setup_logging(LoggingConfig(level="INFO", console_enabled=False))
    emit("info", "module event", component="test")


def test_emitter_requires_valid_level():
    emitter = StructuredLogEmitter("test")
    try:
        emitter.emit("not-a-level", "boom")
    except (ValueError, AttributeError):
        pass
    else:
        raise AssertionError("expected failure")
