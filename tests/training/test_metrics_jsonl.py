"""
tests/training/test_metrics_jsonl.py
====================================
Tests for loss-curve persistence (pre-training gate: loss curves must be
written to disk and be reconstructible, not just printed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadgenesis.training.callbacks import MetricsJsonlCallback, TrainingEvent


def _event(epoch: int, step: int, loss: float, val: float | None = None) -> TrainingEvent:
    metrics = {"loss": loss}
    validation = {"loss": val} if val is not None else {}
    return TrainingEvent(
        epoch=epoch,
        step=step,
        metrics=metrics,
        validation_metrics=validation,
        best_validation_loss=val,
        checkpoint_path=None,
    )


class TestMetricsJsonlCallback:
    def test_writes_one_line_per_event(self, tmp_path: Path):
        cb = MetricsJsonlCallback(str(tmp_path))
        cb.on_train_begin(_event(0, 0, 0.0))
        cb.on_epoch_end(_event(0, 10, 1.5))
        cb.on_validation(_event(0, 10, 1.5, val=1.2))
        cb.on_checkpoint(_event(1, 20, 0.9, val=1.0))
        cb.on_train_end(_event(1, 20, 0.9))

        lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        events = [json.loads(line)["event"] for line in lines]
        assert events == ["train_begin", "epoch_end", "validation", "checkpoint", "train_end"]

    def test_loss_reconstructible(self, tmp_path: Path):
        cb = MetricsJsonlCallback(str(tmp_path))
        for epoch, loss in [(0, 2.0), (1, 1.0), (2, 0.5)]:
            cb.on_epoch_end(_event(epoch, epoch * 10, loss))

        losses = [
            json.loads(line)["metrics"]["loss"]
            for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert losses == [2.0, 1.0, 0.5]

    def test_creates_directory(self, tmp_path: Path):
        target = tmp_path / "nested" / "run"
        MetricsJsonlCallback(str(target)).on_epoch_end(_event(0, 1, 0.1))
        assert (target / "metrics.jsonl").exists()

    def test_validation_metrics_recorded(self, tmp_path: Path):
        cb = MetricsJsonlCallback(str(tmp_path))
        cb.on_validation(_event(3, 30, 0.8, val=0.75))
        row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["event"] == "validation"
        assert row["metrics"]["loss"] == 0.75
        assert row["best_validation_loss"] == 0.75

    def test_custom_filename(self, tmp_path: Path):
        cb = MetricsJsonlCallback(str(tmp_path), filename="run_metrics.jsonl")
        cb.on_epoch_end(_event(0, 1, 0.1))
        assert (tmp_path / "run_metrics.jsonl").exists()