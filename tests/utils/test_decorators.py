"""tests/utils/test_decorators.py"""

from __future__ import annotations

import threading
import time
import warnings

import pytest

from cadgenesis.utils.decorators import (
    deprecated,
    log_calls,
    memoize,
    retry,
    singleton,
    synchronized,
    timed,
)


def test_timed_runs_and_returns_value():
    calls = []

    @timed()
    def add(a, b):
        calls.append(a)
        return a + b

    assert add(1, 2) == 3
    assert calls == [1]


def test_timed_disabled_is_passthrough():
    @timed(enabled=False)
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_retry_succeeds_after_failures():
    attempts = {"n": 0}

    @retry(attempts=3, delay=0.0, backoff=1.0)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_retry_exhausts_and_raises():
    @retry(attempts=2, delay=0.0, backoff=1.0)
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        always_fails()


def test_retry_on_retry_callback():
    seen = []

    @retry(attempts=2, delay=0.0, backoff=1.0, on_retry=lambda n, exc, d: seen.append((n, exc)))
    def flaky():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        flaky()
    assert len(seen) == 1


def test_memoize_caches():
    calls = {"n": 0}

    @memoize
    def square(x):
        calls["n"] += 1
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert square(4) == 16
    assert calls["n"] == 2


def test_memoize_cache_clear():
    @memoize
    def identity(x):
        return x

    identity(1)
    assert len(identity.cache) == 1
    identity.cache_clear()
    assert len(identity.cache) == 0


def test_singleton():
    @singleton
    class Only:
        def __init__(self):
            self.value = 42

    a = Only()
    b = Only()
    assert a is b
    assert a.value == 42


def test_deprecated_warns():
    @deprecated(reason="use new", version="6.0")
    def old():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old() == 1
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_synchronized_serializes_concurrent_access():
    @synchronized
    def increment(counter, lock):
        with lock:
            value = counter["n"]
            time.sleep(0.001)
            counter["n"] = value + 1

    counter = {"n": 0}
    lock = threading.Lock()
    threads = [threading.Thread(target=increment, args=(counter, lock)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert counter["n"] == 8


def test_log_calls_returns_value(caplog):
    @log_calls()
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
