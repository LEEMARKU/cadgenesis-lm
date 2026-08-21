"""tests/utils/test_time.py"""

from __future__ import annotations

import threading
import time

import pytest

from cadgenesis.utils.time import (
    RateLimiter,
    Stopwatch,
    format_duration,
    now_iso,
    now_unix,
    parse_duration,
    utc_now,
)


def test_utc_now_timezone_aware():
    assert utc_now().tzinfo is not None


def test_now_iso():
    assert "T" in now_iso()


def test_now_unix():
    assert now_unix() == pytest.approx(time.time(), abs=1.0)


def test_format_duration():
    assert format_duration(0.4) == "400.0ms"
    assert format_duration(93.2) == "1m 33s"
    assert format_duration(3 * 3600) == "3h 0m"
    assert format_duration(2 * 86400 + 3600) == "2d 1h 0m"


def test_parse_duration():
    assert parse_duration("500ms") == pytest.approx(0.5)
    assert parse_duration("10s") == pytest.approx(10.0)
    assert parse_duration("2m") == pytest.approx(120.0)
    assert parse_duration("1h30m") == pytest.approx(5400.0)
    assert parse_duration("2") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        parse_duration("abc")
    with pytest.raises(ValueError):
        parse_duration("1x")


def test_stopwatch():
    with Stopwatch() as sw:
        time.sleep(0.01)
    assert sw.elapsed >= 0.01
    assert sw.elapsed_str


def test_stopwatch_start_stop():
    sw = Stopwatch()
    sw.start()
    time.sleep(0.01)
    elapsed = sw.stop()
    assert elapsed >= 0.01


def test_rate_limiter_nonblocking():
    limiter = RateLimiter(rate=1.0, burst=3)
    assert limiter.try_acquire()
    assert limiter.try_acquire()
    assert limiter.try_acquire()
    assert not limiter.try_acquire()


def test_rate_limiter_blocks_then_allows():
    limiter = RateLimiter(rate=100.0, burst=1)
    with limiter.acquire():
        pass
    assert not limiter.try_acquire()
    time.sleep(0.03)
    assert limiter.try_acquire()


def test_rate_limiter_invalid_rate():
    with pytest.raises(ValueError):
        RateLimiter(rate=0)


def test_rate_limiter_thread_safety():
    limiter = RateLimiter(rate=0.001, burst=50)
    acquired = []

    def worker():
        if limiter.try_acquire():
            acquired.append(True)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(acquired) <= 50
    assert sum(acquired) > 0
