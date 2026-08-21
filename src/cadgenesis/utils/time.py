"""cadgenesis.utils.time
=====================
Time helpers for CADGenesis-LM v6.0: timestamp formatting, a stopwatch
context manager, duration parsing/formatting, and a token-bucket rate limiter.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator
from datetime import datetime, timezone

from typing_extensions import Self

__all__ = [
    "RateLimiter",
    "Stopwatch",
    "format_duration",
    "monotonic",
    "now_iso",
    "now_unix",
    "parse_duration",
    "sleep",
    "utc_now",
]


def utc_now() -> datetime:
    """Timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """ISO-8601 UTC timestamp string (e.g. ``2026-08-03T12:34:56.123456+00:00``)."""
    return utc_now().isoformat()


def now_unix() -> float:
    """Current Unix time in seconds (float)."""
    return time.time()


def monotonic() -> float:
    """Monotonic clock in seconds (for measuring elapsed time)."""
    return time.monotonic()


def sleep(seconds: float) -> None:
    """Sleep for ``seconds``."""
    time.sleep(seconds)


def format_duration(seconds: float, precision: int = 1) -> str:
    """Human-readable duration string from seconds.

    Examples::

        format_duration(0.4)      -> "400.0ms"
        format_duration(93.2)     -> "1m 33s"
        format_duration(3 * 3600) -> "3h 0m"
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.{precision}f}ms"
    total = round(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_duration(value: str) -> float:
    """Parse a duration string into seconds.

    Supported units: ``ms``, ``s``, ``m``, ``h``, ``d``.  A bare number is
    interpreted as seconds.  Multiple parts may be combined, e.g. ``1h30m``.
    """
    if not value or not isinstance(value, str):
        raise ValueError(f"invalid duration: {value!r}")
    value = value.strip().lower()
    if value.endswith("ms"):
        return float(value[:-2]) / 1000.0
    total = 0.0
    index = 0
    units = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    while index < len(value):
        start = index
        while index < len(value) and (value[index].isdigit() or value[index] == "."):
            index += 1
        if start == index:
            raise ValueError(f"invalid duration: {value!r}")
        number = float(value[start:index])
        if index >= len(value):
            total += number
            break
        unit = value[index]
        if unit not in units:
            raise ValueError(f"unknown duration unit '{unit}' in {value!r}")
        total += number * units[unit]
        index += 1
    return total


class Stopwatch:
    """Context manager measuring elapsed wall time.

    Usage::

        with Stopwatch() as sw:
            work()
        sw.elapsed   # seconds (float)
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
            self._start = None

    def start(self) -> None:
        """Start (or restart) the stopwatch."""
        self._start = time.perf_counter()
        self._elapsed = 0.0

    def stop(self) -> float:
        """Stop and return elapsed seconds."""
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
            self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed seconds (only valid after the context exits or ``stop``)."""
        if self._start is not None:
            return time.perf_counter() - self._start
        return self._elapsed

    @property
    def elapsed_str(self) -> str:
        return format_duration(self.elapsed)


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Usage::

        limiter = RateLimiter(rate=5.0, burst=10)   # 5 tokens/sec, burst 10
        limiter.acquire()                            # blocks until a token is free
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be > 0; got {rate}")
        self.rate = rate
        self.burst = burst if burst is not None else rate
        self._tokens = self.burst
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._updated = now

    @contextlib.contextmanager
    def acquire(self, tokens: float = 1.0) -> Generator[None, None, None]:
        """Block until ``tokens`` are available, then consume them."""
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    break
                missing = tokens - self._tokens
                wait = missing / self.rate
                self._lock.release()
                try:
                    time.sleep(wait)
                finally:
                    self._lock.acquire()
        yield

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking acquire; returns True when tokens were consumed."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
