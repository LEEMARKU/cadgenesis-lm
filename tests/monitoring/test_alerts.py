"""tests/monitoring/test_alerts.py"""

from __future__ import annotations

import pytest

from cadgenesis.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    CallbackAlertHandler,
    LogAlertHandler,
    ThresholdRule,
)


def test_threshold_rule_above():
    rule = ThresholdRule("high_loss", "loss", 5.0, AlertSeverity.WARNING)
    assert not rule.evaluate({"loss": 3.0})
    assert rule.evaluate({"loss": 7.0})
    assert not rule.evaluate({"other": 100})


def test_threshold_rule_below():
    rule = ThresholdRule(
        "low_acc",
        "accuracy",
        0.9,
        severity=AlertSeverity.CRITICAL,
        direction="below",
    )
    assert rule.evaluate({"accuracy": 0.8})
    assert not rule.evaluate({"accuracy": 0.95})


def test_threshold_rule_invalid_direction():
    with pytest.raises(ValueError):
        ThresholdRule("bad", "k", 1.0, direction="sideways")


def test_alert_manager_fires_and_dispatches():
    manager = AlertManager()
    fired = []
    manager.add_handler(CallbackAlertHandler(lambda alert: fired.append(alert)))
    manager.add_rule(ThresholdRule("high", "x", 10.0, AlertSeverity.WARNING))
    alerts = manager.evaluate({"x": 12.0})
    assert len(alerts) == 1
    assert len(fired) == 1
    assert fired[0].rule_name == "high"
    assert fired[0].severity == AlertSeverity.WARNING


def test_alert_manager_duplicate_rule_rejected():
    manager = AlertManager()
    manager.add_rule(ThresholdRule("a", "x", 1.0))
    with pytest.raises(ValueError):
        manager.add_rule(ThresholdRule("a", "y", 2.0))


def test_alert_manager_remove_rule():
    manager = AlertManager()
    manager.add_rule(ThresholdRule("a", "x", 1.0))
    manager.remove_rule("a")
    assert manager.evaluate({"x": 100}) == []


def test_alert_manager_cooldown():
    manager = AlertManager(cooldown_seconds=60.0)
    manager.add_rule(ThresholdRule("a", "x", 1.0))
    assert len(manager.evaluate({"x": 2})) == 1
    assert len(manager.evaluate({"x": 2})) == 0


def test_alert_manager_history():
    manager = AlertManager()
    manager.add_rule(ThresholdRule("a", "x", 1.0))
    manager.evaluate({"x": 2})
    assert len(manager.history) == 1
    manager.clear_history()
    assert manager.history == []


def test_rule_evaluation_exception_skipped():
    def broken(context):
        raise RuntimeError("boom")

    rule = AlertRule("broken", AlertSeverity.INFO, "msg", broken)
    manager = AlertManager()
    manager.add_rule(rule)
    assert manager.evaluate({}) == []


def test_alert_to_dict():
    alert = Alert("rule", AlertSeverity.CRITICAL, "msg", {"k": 1})
    d = alert.to_dict()
    assert d["severity"] == "critical"
    assert d["context"] == {"k": 1}


def test_log_alert_handler_smoke():
    handler = LogAlertHandler()
    handler.handle(Alert("r", AlertSeverity.INFO, "m"))
