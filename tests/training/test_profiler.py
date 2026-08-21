"""tests/training/test_profiler.py"""

from __future__ import annotations

import json
import os
import time

from cadgenesis.training.profiler import ProfilerStats, TrainingProfiler


def test_stats_tokens_per_second():
    stats = ProfilerStats(tokens_processed=1000, total_seconds=2.0)
    assert stats.tokens_per_second() == 500.0


def test_stats_zero_guard():
    stats = ProfilerStats()
    assert stats.tokens_per_second() == 0.0


def test_profiler_times_phases():
    profiler = TrainingProfiler(enabled=True)
    profiler.start()
    with profiler.phase("forward"):
        time.sleep(0.01)
    profiler.step_done(tokens=128)
    summary = profiler.stats
    assert summary.steps == 1
    assert summary.tokens_processed == 128
    assert summary.forward_seconds > 0.0
    assert len(profiler.trace) == 1
    assert profiler.trace[0]["tokens"] == 128.0


def test_profiler_disabled_records_nothing():
    profiler = TrainingProfiler(enabled=False)
    profiler.start()
    with profiler.phase("forward"):
        pass
    profiler.step_done(tokens=128)
    assert profiler.stats.steps == 0
    assert profiler.trace == []


def test_profiler_save_trace(tmp_path):
    profiler = TrainingProfiler(enabled=True)
    profiler.start()
    with profiler.phase("data"):
        pass
    profiler.step_done()
    path = os.path.join(tmp_path, "trace.jsonl")
    profiler.save_trace(path)
    with open(path, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == 1
    assert "data" in lines[0]


def test_profiler_summary_string():
    profiler = TrainingProfiler(enabled=True)
    profiler.start()
    profiler.step_done(tokens=64)
    text = profiler.summary()
    assert "steps=1" in text
    assert "tokens_per_second" in text


def test_profiler_all_phases_accumulate():
    profiler = TrainingProfiler(enabled=True)
    profiler.start()
    with profiler.phase("data"):
        pass
    with profiler.phase("forward"):
        pass
    with profiler.phase("backward"):
        pass
    with profiler.phase("optimizer"):
        pass
    profiler.step_done()
    stats = profiler.stats
    assert stats.data_seconds > 0.0
    assert stats.forward_seconds > 0.0
    assert stats.backward_seconds > 0.0
    assert stats.optimizer_seconds > 0.0
