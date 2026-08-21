"""tests/telemetry/test_logs.py"""

from __future__ import annotations

import pytest

from cadgenesis.telemetry.logs import TelemetryEvent, TelemetryLogger, log_event


def test_telemetry_logger_capture():
    logger = TelemetryLogger("inference", capture=True, buffer_capacity=10)
    logger.info("generated", tokens=32, latency_ms=4.2)
    events = logger.buffer.drain()
    assert len(events) == 1
    assert events[0].level == "info"
    assert events[0].message == "generated"
    assert events[0].fields["tokens"] == 32


def test_telemetry_logger_no_capture():
    logger = TelemetryLogger("inference", capture=False)
    logger.warning("no capture")
    assert logger.buffer.snapshot() == []


def test_telemetry_logger_invalid_level():
    logger = TelemetryLogger("inference")
    with pytest.raises(ValueError):
        logger.log("nope", "x")


def test_event_buffer_bounded():
    logger = TelemetryLogger("x", capture=True, buffer_capacity=3)
    for i in range(5):
        logger.info(f"event {i}")
    events = logger.buffer.snapshot()
    assert len(events) == 3
    assert events[-1].message == "event 4"


def test_event_buffer_clear():
    logger = TelemetryLogger("x", capture=True)
    logger.info("a")
    logger.buffer.clear()
    assert logger.buffer.snapshot() == []


def test_all_level_methods():
    logger = TelemetryLogger("x", capture=True)
    logger.debug("d")
    logger.info("i")
    logger.warning("w")
    logger.error("e")
    logger.critical("c")
    assert len(logger.buffer.snapshot()) == 5


def test_telemetry_event_to_dict():
    event = TelemetryEvent("info", "m", {"k": "v"})
    assert event.to_dict() == {"level": "info", "message": "m", "k": "v"}


def test_module_log_event_smoke():
    log_event("info", "module event", a=1)
