"""tests/monitoring/test_health.py"""

from __future__ import annotations

import pytest

from cadgenesis.monitoring.health import (
    HealthChecker,
    HealthResult,
    HealthStatus,
    check_disk_usage,
    check_memory_usage,
)


def test_health_result_status():
    assert HealthResult("a", True).status == HealthStatus.HEALTHY
    assert HealthResult("a", False).status == HealthStatus.UNHEALTHY


def test_checker_registration():
    checker = HealthChecker()
    checker.register("ok", lambda: HealthResult("ok", True, "fine"))
    assert checker.names() == ["ok"]
    checker.unregister("ok")
    assert checker.names() == []


def test_checker_duplicate_registration():
    checker = HealthChecker()
    checker.register("a", lambda: HealthResult("a", True))
    with pytest.raises(ValueError):
        checker.register("a", lambda: HealthResult("a", True))


def test_checker_run_summary_healthy():
    checker = HealthChecker()
    checker.register("a", lambda: HealthResult("a", True))
    checker.register("b", lambda: HealthResult("b", True))
    summary = checker.summary()
    assert summary["status"] == "healthy"
    assert summary["healthy"] == 2
    assert summary["total"] == 2


def test_checker_run_summary_degraded():
    checker = HealthChecker()
    checker.register("a", lambda: HealthResult("a", True))
    checker.register("b", lambda: HealthResult("b", False))
    summary = checker.summary()
    assert summary["status"] == "degraded"


def test_checker_run_summary_unhealthy():
    checker = HealthChecker()
    checker.register("a", lambda: HealthResult("a", False))
    summary = checker.summary()
    assert summary["status"] == "unhealthy"


def test_checker_captures_exceptions():
    checker = HealthChecker()

    def boom():
        raise RuntimeError("x")

    checker.register("bad", boom)
    results = checker.run()
    assert len(results) == 1
    assert not results[0].ok


def test_checker_empty_summary():
    summary = HealthChecker().summary()
    assert summary["status"] == "healthy"
    assert summary["total"] == 0


def test_check_memory_and_disk_smoke():
    result = check_memory_usage()
    assert isinstance(result, HealthResult)
    disk = check_disk_usage(".")
    assert isinstance(disk, HealthResult)
