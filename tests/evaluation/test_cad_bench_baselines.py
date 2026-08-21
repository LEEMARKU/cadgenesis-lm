"""
tests/evaluation/test_cad_bench_baselines.py
============================================
Tests for benchmark baselines + report generation (pre-training gate:
benchmarks must be comparable against non-model baselines).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadgenesis.evaluation.cad_bench import (
    CADBenchItem,
    CADBenchmark,
    FrequencyBaseline,
    RandomBaseline,
    write_benchmark_report,
)
from cadgenesis.distillation.rlvr import MockOracle


def _items(n: int = 4, seed: int = 1) -> list[CADBenchItem]:
    return [
        CADBenchItem(
            prompt=f"prompt {i}",
            reference_ids=list(range(5)),
        )
        for i in range(n)
    ]


class TestRandomBaseline:
    def test_length_within_bounds(self):
        base = RandomBaseline(vocab_size=50, seed=7, min_len=4)
        for _ in range(20):
            ids = base.sample("x")
            assert 4 <= len(ids) <= 64

    def test_ids_in_vocab_range(self):
        base = RandomBaseline(vocab_size=10, seed=7)
        assert all(0 <= i < 10 for i in base.sample("x", max_len=32))

    def test_deterministic_with_seed(self):
        a = RandomBaseline(vocab_size=50, seed=3).sample("p", max_len=20)
        b = RandomBaseline(vocab_size=50, seed=3).sample("p", max_len=20)
        assert a == b


class TestFrequencyBaseline:
    def test_cycles_top_k(self):
        base = FrequencyBaseline(token_ids=[5, 6, 7], top_k=2)
        ids = base.sample("p", max_len=6)
        assert ids == [5, 6, 5, 6, 5, 6]

    def test_top_k_clamped(self):
        base = FrequencyBaseline(token_ids=[1, 2], top_k=99)
        assert base.sample("p", max_len=4) == [1, 2, 1, 2]

    def test_empty_token_ids_rejected(self):
        with pytest.raises(ValueError):
            FrequencyBaseline(token_ids=[])


class TestEvaluateBaseline:
    def test_mock_oracle_random_zero(self):
        items = [CADBenchItem(prompt="p", reference_ids=[1, 2, 3])]
        oracle = MockOracle(valid_ids=[1, 2, 3])
        bench = CADBenchmark(items=items, oracle=oracle)
        result = bench.evaluate_baseline(RandomBaseline(vocab_size=1000, seed=0), max_len=3)
        assert result.compile_rate == 0.0
        assert result.oracle_avg_reward == 0.0
        assert result.exact_match == 0.0

    def test_frequency_baseline_matches(self):
        items = [CADBenchItem(prompt="p", reference_ids=[1, 2, 3])]
        oracle = MockOracle(valid_ids=[1, 2, 3])
        bench = CADBenchmark(items=items, oracle=oracle)
        result = bench.evaluate_baseline(FrequencyBaseline(token_ids=[1, 2, 3], top_k=3), max_len=3)
        assert result.oracle_avg_reward == 1.0
        assert result.exact_match == 1.0
        assert result.compile_rate == 1.0

    def test_mean_confidence_zero_for_baselines(self):
        bench = CADBenchmark(items=_items(), oracle=MockOracle())
        result = bench.evaluate_baseline(FrequencyBaseline(token_ids=[1]), max_len=8)
        assert result.mean_confidence == 0.0


class TestWriteBenchmarkReport:
    def test_report_roundtrip(self, tmp_path: Path):
        items = _items(2)
        oracle = MockOracle(valid_ids=list(range(5)))
        bench = CADBenchmark(items=items, oracle=oracle)
        a = bench.evaluate_baseline(FrequencyBaseline(token_ids=[0, 1, 2, 3, 4], top_k=5), max_len=5)
        b = bench.evaluate_baseline(RandomBaseline(vocab_size=10, seed=0), max_len=5)
        out = tmp_path / "report.md"
        text = write_benchmark_report(
            str(out),
            [("freq", a), ("random", b)],
            title="Bench Test",
        )
        assert out.exists()
        assert "Bench Test" in text
        assert "freq" in text and "random" in text
        for line in text.splitlines():
            assert "junk" not in line  # sanity: rendered content only

    def test_report_aggregate_best(self, tmp_path: Path):
        items = _items(2)
        oracle = MockOracle(valid_ids=list(range(5)))
        bench = CADBenchmark(items=items, oracle=oracle)
        good = bench.evaluate_baseline(FrequencyBaseline(token_ids=[0, 1, 2, 3, 4], top_k=5), max_len=5)
        bad = bench.evaluate_baseline(RandomBaseline(vocab_size=10, seed=0), max_len=5)
        out = tmp_path / "report.md"
        write_benchmark_report(str(out), [("random", bad), ("freq", good)])
        raw = out.read_text(encoding="utf-8")
        assert "freq" in raw and "random" in raw
