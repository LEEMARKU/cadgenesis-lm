"""
tests/evaluation/test_cad_bench.py
==================================
Tests for the CAD-specific evaluation benchmark.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.distillation.rlvr import MockOracle
from cadgenesis.evaluation.cad_bench import (
    CADBenchItem,
    CADBenchmark,
    CADBenchResult,
)
from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


@pytest.fixture
def engine():
    torch.manual_seed(0)
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box"])
    model = GeometryAwareTransformer(CADConfig.mini())
    return CADInferenceEngine(model, tok, device="cpu")


class TestSequenceAccuracy:
    def test_perfect_match(self):
        assert CADBenchmark._sequence_accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_partial_match(self):
        assert CADBenchmark._sequence_accuracy([1, 2, 4], [1, 2, 3]) == pytest.approx(2 / 3)

    def test_empty_reference(self):
        assert CADBenchmark._sequence_accuracy([1, 2], []) == 0.0


class TestCADBenchmark:
    def test_requires_items(self):
        with pytest.raises(ValueError, match="at least one item"):
            CADBenchmark([])

    def test_metrics_with_mock_oracle(self, engine):
        torch.manual_seed(1)
        target = engine.sample("create a steel box", max_len=8, temperature=0.0).ids
        oracle = MockOracle(valid_ids=target)
        bench = CADBenchmark(
            [CADBenchItem(prompt="create a steel box", reference_ids=target)],
            oracle=oracle,
        )
        res = bench.evaluate(engine, max_len=8, temperature=0.0)
        assert isinstance(res, CADBenchResult)
        assert res.num_samples == 1
        assert res.compile_rate == 1.0
        assert res.exact_match == 1.0
        assert res.sequence_accuracy == 1.0
        assert res.oracle_avg_reward == 1.0
        assert 0.0 <= res.mean_confidence <= 1.0

    def test_constraint_checker(self, engine):
        def checker(ids):
            return ids == engine.sample("create a steel box", max_len=8, temperature=0.0).ids

        bench = CADBenchmark(
            [CADBenchItem(prompt="create a steel box")],
            oracle=MockOracle(),
            constraint_checker=checker,
        )
        res = bench.evaluate(engine, max_len=8, temperature=0.0)
        assert res.constraint_sat_rate == 1.0

    def test_failing_oracle_scores_zero(self, engine):
        bench = CADBenchmark(
            [CADBenchItem(prompt="create a steel box", reference_ids=[])],
            oracle=MockOracle(valid_ids=[9999]),
        )
        res = bench.evaluate(engine, max_len=8, temperature=0.0)
        assert res.compile_rate == 0.0
        assert res.exact_match == 0.0
        assert res.sequence_accuracy == 0.0
        assert res.oracle_avg_reward == 0.0

    def test_summary_string(self):
        res = CADBenchResult(1.0, 0.5, 0.25, 0.6, 0.9, 0.7, 2)
        summary = res.summary()
        assert "compile_rate=1.000" in summary
        assert "reward=0.700" in summary
        assert "(n=2)" in summary
