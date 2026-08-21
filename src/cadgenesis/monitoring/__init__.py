"""cadgenesis.monitoring
=====================
Health checks, drift detection, and alerting for CADGenesis-LM v6.0.
"""

from cadgenesis.monitoring.alerts import (
    Alert,
    AlertContext,
    AlertHandler,
    AlertManager,
    AlertRule,
    AlertSeverity,
    CallbackAlertHandler,
    LogAlertHandler,
    ThresholdRule,
)
from cadgenesis.monitoring.drift import (
    DriftMetric,
    DriftReport,
    FeatureDriftMonitor,
    compute_drift,
)
from cadgenesis.monitoring.health import (
    HealthChecker,
    HealthResult,
    HealthStatus,
    check_disk_usage,
    check_memory_usage,
)

__all__ = [
    "Alert",
    "AlertContext",
    "AlertHandler",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "CallbackAlertHandler",
    "DriftMetric",
    "DriftReport",
    "FeatureDriftMonitor",
    "HealthChecker",
    "HealthResult",
    "HealthStatus",
    "LogAlertHandler",
    "ThresholdRule",
    "check_disk_usage",
    "check_memory_usage",
    "compute_drift",
]
