"""tests/training/test_metrics.py"""

from __future__ import annotations

import pytest

from cadgenesis.training.metrics import MetricsTracker, compute_accuracy, log_summary


def test_tracker_starts_empty():
    tracker = MetricsTracker()
    assert tracker.steps == 0
    assert tracker.average_loss() == 0.0
    assert tracker.accuracy() == 0.0


def test_tracker_update_accumulates():
    tracker = MetricsTracker()
    tracker.update(loss=2.0, correct_tokens=3, total_tokens=10)
    tracker.update(loss=4.0, correct_tokens=7, total_tokens=10)
    assert tracker.steps == 2
    assert tracker.average_loss() == 3.0
    assert tracker.accuracy() == 0.5


def test_tracker_ema_loss():
    tracker = MetricsTracker(ema_alpha=0.5)
    tracker.update(loss=1.0)
    tracker.update(loss=3.0)
    assert tracker.ema_loss == 2.0


def test_tracker_perplexity():
    tracker = MetricsTracker()
    tracker.update(loss=2.0)
    assert tracker.perplexity() > 7.0


def test_tracker_snapshot_keys():
    tracker = MetricsTracker()
    tracker.update(loss=1.0, correct_tokens=9, total_tokens=10)
    snapshot = tracker.snapshot()
    assert snapshot["steps"] == 1.0
    assert snapshot["loss"] == 1.0
    assert snapshot["accuracy"] == 0.9
    assert "perplexity" in snapshot


def test_tracker_record_epoch_and_reset():
    tracker = MetricsTracker()
    tracker.update(loss=1.0)
    tracker.record_epoch("train")
    assert len(tracker.history) == 1
    assert tracker.history[0]["tag"] == "train"
    tracker.reset()
    assert tracker.steps == 0
    assert tracker.ema_loss is None


def test_compute_accuracy():
    assert compute_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert compute_accuracy([1, 2, 3], [1, 9, 3]) == pytest.approx(2 / 3)
    assert compute_accuracy([], []) == 0.0


def test_log_summary_skips_containers():
    metrics = {"loss": 0.5, "history": [1, 2, 3]}
    line = log_summary(metrics)
    assert "loss=0.5" in line
    assert "history" not in line
