"""cadgenesis.monitoring.alerts
============================
Alerting framework for CADGenesis-LM v6.0: severity levels, rules, and
dispatch handlers.

Rules evaluate a context (dict) and, when they fire, produce an ``Alert`` that
is dispatched to registered ``AlertHandler`` instances.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single fired alert."""

    rule_name: str
    severity: AlertSeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    fired_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "fired_at": self.fired_at,
        }


AlertContext = dict[str, Any]


@dataclass
class AlertRule:
    """Predicate-based alert rule.

    ``evaluate`` returns True when the alert should fire for the context.
    """

    name: str
    severity: AlertSeverity
    message: str
    predicate: Callable[[AlertContext], bool]

    def evaluate(self, context: AlertContext) -> bool:
        try:
            return bool(self.predicate(context))
        except Exception:
            log.exception("alert rule %s raised while evaluating", self.name)
            raise


class AlertHandler:
    """Base class for alert sinks."""

    def handle(self, alert: Alert) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LogAlertHandler(AlertHandler):
    """Writes alerts to the Python logging system."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("cadgenesis.monitoring.alerts")

    def handle(self, alert: Alert) -> None:
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
        }[alert.severity]
        self.logger.log(level, "[%s] %s: %s", alert.severity.value, alert.rule_name, alert.message)


class CallbackAlertHandler(AlertHandler):
    """Dispatches alerts to an arbitrary callback."""

    def __init__(self, callback: Callable[[Alert], Any]) -> None:
        self.callback = callback

    def handle(self, alert: Alert) -> None:
        self.callback(alert)


class AlertManager:
    """Registers rules and handlers, evaluates contexts, and dispatches alerts.

    Usage::

        manager = AlertManager()
        manager.add_handler(LogAlertHandler())
        manager.add_rule(ThresholdRule("high_loss", "loss", 5.0, AlertSeverity.WARNING))
        fired = manager.evaluate({"loss": 7.0})
    """

    def __init__(self, cooldown_seconds: float = 0.0) -> None:
        self._rules: list[AlertRule] = []
        self._handlers: list[AlertHandler] = []
        self._history: list[Alert] = []
        self._last_fired: dict[str, float] = {}
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            if any(r.name == rule.name for r in self._rules):
                raise ValueError(f"alert rule '{rule.name}' already registered")
            self._rules.append(rule)

    def remove_rule(self, name: str) -> None:
        with self._lock:
            self._rules = [r for r in self._rules if r.name != name]

    def add_handler(self, handler: AlertHandler) -> None:
        with self._lock:
            self._handlers.append(handler)

    def clear_history(self) -> None:
        with self._lock:
            self._history = []

    @property
    def history(self) -> list[Alert]:
        with self._lock:
            return list(self._history)

    def evaluate(self, context: AlertContext) -> list[Alert]:
        """Evaluate all rules against ``context``; fire and dispatch matches."""
        with self._lock:
            rules = list(self._rules)
            handlers = list(self._handlers)
        fired: list[Alert] = []
        now = time.monotonic()
        for rule in rules:
            try:
                matches = rule.evaluate(context)
            except Exception:
                log.exception("rule %s evaluation failed", rule.name)
                continue
            if not matches:
                continue
            last = self._last_fired.get(rule.name)
            if last is not None and now - last < self.cooldown_seconds:
                continue
            self._last_fired[rule.name] = now
            alert = Alert(rule.name, rule.severity, rule.message, dict(context))
            with self._lock:
                self._history.append(alert)
            fired.append(alert)
            for handler in handlers:
                try:
                    handler.handle(alert)
                except Exception:  # noqa: PERF203
                    log.exception("alert handler failed for rule %s", rule.name)
        return fired


def ThresholdRule(
    name: str,
    context_key: str,
    threshold: float,
    severity: AlertSeverity = AlertSeverity.WARNING,
    message: str | None = None,
    direction: str = "above",
) -> AlertRule:
    """Factory building a numeric threshold rule.

    Args:
        name: Unique rule name.
        context_key: Key in the context dict holding the numeric value.
        threshold: Comparison threshold.
        severity: Severity of the fired alert.
        message: Alert message (defaults to an auto-generated one).
        direction: "above" fires when value > threshold; "below" when value < threshold.
    """
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below'; got {direction!r}")

    def predicate(context: AlertContext) -> bool:
        value = context.get(context_key)
        if value is None:
            return False
        return value > threshold if direction == "above" else value < threshold

    return AlertRule(
        name=name,
        severity=severity,
        message=message
        or f"threshold {'exceeded' if direction == 'above' else 'fell below'} "
        f"for {context_key} ({direction} {threshold})",
        predicate=predicate,
    )
