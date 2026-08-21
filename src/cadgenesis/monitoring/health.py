"""cadgenesis.monitoring.health
============================
Readiness/liveness health-check framework for CADGenesis-LM v6.0.

A health check is any callable returning a ``HealthResult``.  The
``HealthChecker`` runs registered checks, aggregates them, and produces a
machine-readable summary suitable for serving endpoints.
"""

from __future__ import annotations

import enum
import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class HealthStatus(str, enum.Enum):
    """Aggregate health classification."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthResult:
    """Outcome of a single health check."""

    name: str
    ok: bool
    detail: str = ""
    checked_at: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

    @property
    def status(self) -> HealthStatus:
        return HealthStatus.HEALTHY if self.ok else HealthStatus.UNHEALTHY


HealthCheckFn = Callable[[], HealthResult]


class HealthChecker:
    """Runs registered health checks and aggregates their results.

    Usage::

        checker = HealthChecker()
        checker.register("model", lambda: HealthResult("model", True, "loaded"))
        summary = checker.run()
    """

    def __init__(self) -> None:
        self._checks: dict[str, HealthCheckFn] = {}
        self._lock = threading.Lock()

    def register(self, name: str, check: HealthCheckFn) -> None:
        """Register a named health check."""
        if not name:
            raise ValueError("health check name must not be empty")
        with self._lock:
            if name in self._checks:
                raise ValueError(f"health check '{name}' already registered")
            self._checks[name] = check

    def unregister(self, name: str) -> None:
        with self._lock:
            self._checks.pop(name, None)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._checks)

    def run(self) -> list[HealthResult]:
        """Execute every registered check; failures are captured as results."""
        with self._lock:
            checks = dict(self._checks)
        results: list[HealthResult] = []
        for name, check in checks.items():
            try:
                result = check()
                if not isinstance(result, HealthResult):
                    result = HealthResult(name, bool(result), f"non-standard result {result!r}")
            except Exception as exc:
                log.exception("health check %s raised", name)
                result = HealthResult(name, False, f"raised {exc!r}")
            results.append(result)
        return results

    def summary(self, results: list[HealthResult] | None = None) -> dict:
        """Aggregate into a dict with an overall status."""
        results = results if results is not None else self.run()
        if not results:
            return {"status": HealthStatus.HEALTHY.value, "checks": [], "healthy": 0, "total": 0}
        healthy = sum(1 for r in results if r.ok)
        total = len(results)
        if healthy == total:
            status = HealthStatus.HEALTHY
        elif healthy == 0:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.DEGRADED
        return {
            "status": status.value,
            "checks": [vars(r) for r in results],
            "healthy": healthy,
            "total": total,
            "checked_at": time.time(),
        }


def check_memory_usage(threshold_fraction: float = 0.9) -> HealthResult:
    """Health check for system memory utilisation."""

    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return HealthResult(
            "memory",
            True,
            "psutil not installed; memory check skipped",
            data={"available": False},
        )
    usage = psutil.virtual_memory()
    fraction = usage.used / usage.total
    ok = fraction < threshold_fraction
    return HealthResult(
        "memory",
        ok,
        f"used {fraction * 100:.1f}% of {usage.total / 1e9:.1f} GiB",
        data={"used_bytes": usage.used, "total_bytes": usage.total, "fraction": fraction},
    )


def check_disk_usage(path: str = ".", threshold_fraction: float = 0.9) -> HealthResult:
    """Health check for disk space on the given path."""

    usage = shutil.disk_usage(path)
    fraction = usage.used / usage.total
    ok = fraction < threshold_fraction
    return HealthResult(
        "disk",
        ok,
        f"{path}: used {fraction * 100:.1f}%",
        data={
            "path": path,
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "fraction": fraction,
        },
    )
