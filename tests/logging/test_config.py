"""tests/logging/test_config.py"""

from __future__ import annotations

import json
import logging

import pytest

from cadgenesis.logging.config import LoggingConfig, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _clean_logger():
    yield
    root = logging.getLogger("cadgenesis")
    root.handlers = []
    root.disabled = False


def test_logging_config_validates_level():
    assert LoggingConfig(level="DEBUG").level == "DEBUG"
    with pytest.raises(ValueError):
        LoggingConfig(level="NOT_A_LEVEL")


def test_setup_logging_returns_logger():
    logger = setup_logging(LoggingConfig(level="DEBUG", console_enabled=False))
    assert logger.name == "cadgenesis"
    assert logger.level == logging.DEBUG


def test_setup_logging_console_handler(capsys):
    setup_logging(LoggingConfig(level="INFO", console_enabled=True))
    logger = logging.getLogger("cadgenesis")
    logger.info("hello console")
    out = capsys.readouterr().out
    assert "hello console" in out


def test_setup_logging_file_handler(tmp_path):
    path = str(tmp_path / "app.log")
    setup_logging(
        LoggingConfig(
            level="INFO",
            console_enabled=False,
            file_enabled=True,
            file_path=path,
        )
    )
    logging.getLogger("cadgenesis").warning("to file")
    assert "to file" in (tmp_path / "app.log").read_text(encoding="utf-8")


def test_setup_logging_json_format(tmp_path):
    path = str(tmp_path / "app.jsonl")
    setup_logging(
        LoggingConfig(
            level="INFO",
            console_enabled=False,
            file_enabled=True,
            file_path=path,
            json_format=True,
        )
    )
    logging.getLogger("cadgenesis").warning("structured", extra={"step": 3})
    lines = (tmp_path / "app.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    record = json.loads(lines[0])
    assert record["level"] == "WARNING"
    assert record["message"] == "structured"
    assert record["step"] == 3


def test_get_logger_namespaced():
    logger = get_logger("my.sub")
    assert logger.name == "cadgenesis.my.sub"


def test_get_logger_accepts_full_name():
    logger = get_logger("cadgenesis.foo")
    assert logger.name == "cadgenesis.foo"
