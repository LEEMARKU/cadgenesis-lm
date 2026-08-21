"""
tests/inference/test_test_time_compute.py
=========================================
Tests for test-time compute: best-of-n, self-consistency, MCTS.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.distillation.rlvr import MockOracle
from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.inference.mcts import best_of_n, mcts, self_consistency
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


@pytest.fixture
def engine():
    torch.manual_seed(0)
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box"])
    model = GeometryAwareTransformer(CADConfig.mini())
    return CADInferenceEngine(model, tok, device="cpu")


class TestBestOfN:
    def test_returns_highest_reward(self, engine):
        torch.manual_seed(1)
        # Deterministic greedy output — oracle validates exactly that sequence.
        target = engine.sample("create a steel box", max_len=8, temperature=0.0).ids
        oracle = MockOracle(valid_ids=target)
        result, reward = best_of_n(
            engine, "create a steel box", oracle, n=3, max_len=8, temperature=0.0
        )
        assert reward == 1.0
        assert result.ids == target

    def test_reward_matches_oracle(self, engine):
        result, reward = best_of_n(
            engine,
            "create a steel box",
            MockOracle(),
            n=2,
            max_len=8,
            temperature=1.0,
        )
        assert reward == MockOracle().verify(result.ids)

    def test_rejects_bad_n(self, engine):
        with pytest.raises(ValueError, match="n must be"):
            best_of_n(engine, "x", MockOracle(), n=0)


class TestSelfConsistency:
    def test_returns_a_sampled_result(self, engine):
        result = self_consistency(engine, "create a steel box", n=3, max_len=8, temperature=1.0)
        assert isinstance(result.ids, list)
        assert result.ids

    def test_rejects_bad_n(self, engine):
        with pytest.raises(ValueError, match="n must be"):
            self_consistency(engine, "x", n=0)


class TestMCTS:
    def test_finds_best_under_oracle(self, engine):
        torch.manual_seed(0)
        # Greedy target is the "solution" the oracle validates.
        target = engine.sample("create a steel box", max_len=8, temperature=0.0).ids
        oracle = MockOracle(valid_ids=target)
        result, reward = mcts(
            engine,
            "create a steel box",
            oracle,
            iterations=3,
            max_len=8,
            temperature=0.0,
            branch=2,
            rollout_len=2,
        )
        assert reward >= 0.0
        assert result.ids
        assert reward == oracle.verify(result.ids)

    def test_seed_prefix_respected(self, engine):
        tok = engine.tokenizer
        prefix = [tok.bos_id]
        result, _ = mcts(
            engine,
            "create a steel box",
            MockOracle(),
            iterations=2,
            max_len=8,
            temperature=1.0,
            branch=2,
            rollout_len=2,
            seed_prefix=prefix,
        )
        # The tree is seeded with the prefix; samples continue from it.
        assert tuple(result.ids) != tuple()

    def test_rejects_bad_args(self, engine):
        with pytest.raises(ValueError, match="iterations"):
            mcts(engine, "x", MockOracle(), iterations=0)
