"""System Health Monitoring - Model, memory, agents, inference, APIs, GPUs, simulations,
workloads."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Thread
from typing import Any

import psutil


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """A single health metric."""

    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float | None = None
    threshold_critical: float | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Complete system health report."""

    report_id: str
    overall_status: HealthStatus
    metrics: list[HealthMetric] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemHealthMonitor:
    """Monitors system health across all components."""

    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._custom_checks: dict[str, Callable[[], list[HealthMetric]]] = {}
        self._reports: dict[str, HealthReport] = {}
        self._running = False
        self._monitor_thread: Thread | None = None
        self._lock = RLock()

    def register_check(self, name: str, check_fn: Callable[[], list[HealthMetric]]) -> None:
        """Register a custom health check."""
        with self._lock:
            self._custom_checks[name] = check_fn

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def _monitor_loop(self) -> None:
        while self._running:
            self.check_health()
            time.sleep(self.check_interval)

    def check_health(self) -> HealthReport:
        """Run all health checks and generate report."""
        metrics = []

        # System metrics
        metrics.extend(self._check_system())

        # GPU metrics (if available)
        metrics.extend(self._check_gpu())

        # Custom checks
        for name, check_fn in self._custom_checks.items():
            try:
                metrics.extend(check_fn())
            except Exception as e:
                metrics.append(
                    HealthMetric(
                        name=f"{name}_error",
                        value=0,
                        unit="",
                        status=HealthStatus.UNHEALTHY,
                        metadata={"error": str(e)},
                    )
                )

        # Determine overall status
        statuses = [m.status for m in metrics]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # Generate alerts
        alerts = []
        for m in metrics:
            if m.status == HealthStatus.UNHEALTHY:
                alerts.append(f"CRITICAL: {m.name} = {m.value} {m.unit}")
            elif m.status == HealthStatus.DEGRADED:
                alerts.append(f"WARNING: {m.name} = {m.value} {m.unit}")

        report = HealthReport(
            report_id=str(uuid.uuid4()),
            overall_status=overall,
            metrics=metrics,
            alerts=alerts,
        )

        with self._lock:
            self._reports[report.report_id] = report

        return report

    def _check_system(self) -> list[HealthMetric]:
        metrics = []

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(
            HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status=HealthStatus.HEALTHY
                if cpu_percent < 80
                else (HealthStatus.DEGRADED if cpu_percent < 95 else HealthStatus.UNHEALTHY),
                threshold_warning=80,
                threshold_critical=95,
            )
        )

        # Memory
        mem = psutil.virtual_memory()
        metrics.append(
            HealthMetric(
                name="memory_usage",
                value=mem.percent,
                unit="%",
                status=HealthStatus.HEALTHY
                if mem.percent < 80
                else (HealthStatus.DEGRADED if mem.percent < 95 else HealthStatus.UNHEALTHY),
                threshold_warning=80,
                threshold_critical=95,
            )
        )

        # Disk
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(
            HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="%",
                status=HealthStatus.HEALTHY
                if disk_percent < 80
                else (HealthStatus.DEGRADED if disk_percent < 95 else HealthStatus.UNHEALTHY),
                threshold_warning=80,
                threshold_critical=95,
            )
        )

        return metrics

    def _check_gpu(self) -> list[HealthMetric]:
        metrics = []
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                    metrics.append(
                        HealthMetric(
                            name=f"gpu_{i}_memory_allocated",
                            value=allocated,
                            unit="GB",
                            status=HealthStatus.HEALTHY,
                        )
                    )
                    metrics.append(
                        HealthMetric(
                            name=f"gpu_{i}_memory_reserved",
                            value=reserved,
                            unit="GB",
                            status=HealthStatus.HEALTHY
                            if reserved < total * 0.9
                            else HealthStatus.DEGRADED,
                            threshold_warning=total * 0.8,
                            threshold_critical=total * 0.95,
                        )
                    )
                    metrics.append(
                        HealthMetric(
                            name=f"gpu_{i}_memory_utilization",
                            value=(reserved / total) * 100,
                            unit="%",
                            status=HealthStatus.HEALTHY
                            if reserved < total * 0.9
                            else HealthStatus.DEGRADED,
                        )
                    )
        except Exception:
            pass
        return metrics

    def get_latest_report(self) -> HealthReport | None:
        with self._lock:
            if not self._reports:
                return None
            return max(self._reports.values(), key=lambda r: r.timestamp)

    def get_report(self, report_id: str) -> HealthReport | None:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self, limit: int = 100) -> list[HealthReport]:
        with self._lock:
            reports = sorted(self._reports.values(), key=lambda r: r.timestamp, reverse=True)
            return reports[:limit]
