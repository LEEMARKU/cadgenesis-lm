"""
tests/training/test_rlvr_pipeline.py
====================================
Tests for the closed-loop RLVR orchestration surface.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.distillation.rlvr import MockOracle
from cadgenesis.evaluation.cad_bench import CADBenchResult
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.training.rlvr_pipeline import RLVRPipeline

PROMPTS = ["create a steel box"]


@pytest.fixture
def pipe() -> RLVRPipeline:
    torch.manual_seed(0)
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(PROMPTS)
    return RLVRPipeline.from_config(
        CADConfig.mini(),
        tok,
        device="cpu",
        oracle=MockOracle(),
        lr=1e-4,
        num_generations=2,
        max_gen_len=8,
    )


class TestRLVRPipeline:
    def test_builds_engine_and_trainer(self, pipe):
        assert pipe.engine is not None
        assert pipe.trainer.grpo is not None
        assert pipe.max_gen_len == 8

    def test_train_returns_stats(self, pipe):
        stats = pipe.train(PROMPTS, steps=1, temperature=1.0)
        assert {"loss", "mean_reward", "mean_kl", "std_reward"} <= set(stats)
        assert torch.isfinite(torch.tensor(stats["loss"]))

    def test_evaluate_returns_result(self, pipe):
        res = pipe.evaluate(PROMPTS, temperature=0.0)
        assert isinstance(res, CADBenchResult)
        assert res.num_samples == 1

    def test_evaluate_with_items(self, pipe):
        from cadgenesis.evaluation.cad_bench import CADBenchItem

        res = pipe.evaluate(items=[CADBenchItem(prompt=PROMPTS[0])], temperature=0.0)
        assert res.num_samples == 1

    def test_test_time_compute(self, pipe):
        best, reward = pipe.best_of_n(PROMPTS[0], n=2, temperature=1.0)
        assert best.ids
        assert reward == 0.0 or reward == 1.0
        sc = pipe.self_consistency(PROMPTS[0], n=2, temperature=1.0)
        assert sc.ids
        mcts_result, mcts_reward = pipe.mcts(
            PROMPTS[0], iterations=2, temperature=1.0, branch=2, rollout_len=2
        )
        assert mcts_result.ids
        assert mcts_reward in (0.0, 1.0)

    def test_eagle_train_and_speculative(self, pipe):
        tok = pipe.tokenizer
        reference = [
            tok.vocab["SKETCH_RECT"],
            tok.vocab["EXTRUDE"],
            tok.vocab["BOX"],
            tok.eos_id,
        ]
        head = pipe.train_eagle([reference], steps=5, lr=1e-3)
        greedy = pipe.engine.greedy(PROMPTS[0], max_len=8, use_cache=True)
        speculative = pipe.speculative(PROMPTS[0], head, max_len=8, num_speculative_tokens=2)
        assert speculative.ids == greedy.ids

    def test_logprob_edge_cases(self, pipe):
        # A one-token completion has no shifted target -> zero scalar.
        lp = pipe._logprob_fn(pipe.model, pipe.tokenizer.encode_text(PROMPTS[0]), [7])
        assert lp.shape == ()
        assert float(lp) == 0.0
        # An empty completion is also a zero scalar.
        lp2 = pipe._logprob_fn(pipe.model, pipe.tokenizer.encode_text(PROMPTS[0]), [])
        assert float(lp2) == 0.0
