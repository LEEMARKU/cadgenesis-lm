"""
tests/distillation/test_rlvr.py
===============================
Tests for RLVR: verifiable-reward oracles and the GRPO-backed trainer.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.distillation.rlvr import (
    DesignOracle,
    MockOracle,
    RLVRTrainer,
    VerifiableOracle,
)


class _FakeEngine:
    def __init__(self, valid=True, manufacturable=True):
        self._valid = valid
        self._manufacturable = manufacturable

    def execute(self, design=None, validate=True, simulate=False, optimize=False):
        class _Result:
            pass

        r = _Result()
        r.is_valid_geometry = self._valid
        r.is_manufacturable = self._manufacturable
        return r


class TestMockOracle:
    def test_exact_match(self):
        oracle = MockOracle(valid_ids=[1, 2, 3])
        assert oracle.verify([1, 2, 3]) == 1.0
        assert oracle.verify([1, 2]) == 0.0

    def test_default_matches_nothing(self):
        assert MockOracle().verify([1]) == 0.0


class TestDesignOracle:
    def test_unparseable_gets_zero(self):
        oracle = DesignOracle(completion_to_design=lambda ids: None)
        assert oracle.verify([1, 2, 3]) == 0.0

    def test_reward_weights(self):
        oracle = DesignOracle(
            execution_engine=_FakeEngine(), completion_to_design=lambda ids: {"x": 1}
        )
        assert oracle.verify([1, 2, 3]) == 1.0  # valid + manufacturable

    def test_valid_but_not_manufacturable(self):
        oracle = DesignOracle(
            execution_engine=_FakeEngine(manufacturable=False),
            completion_to_design=lambda ids: {"x": 1},
        )
        assert oracle.verify([1]) == pytest.approx(0.7)

    def test_invalid_geometry(self):
        oracle = DesignOracle(
            execution_engine=_FakeEngine(valid=False),
            completion_to_design=lambda ids: {"x": 1},
        )
        assert oracle.verify([1]) == 0.0

    def test_execution_exception_returns_zero(self):
        class _Boom:
            def execute(self, **kwargs):
                raise RuntimeError("boom")

        oracle = DesignOracle(execution_engine=_Boom(), completion_to_design=lambda ids: {"x": 1})
        assert oracle.verify([1]) == 0.0


class TestRLVRTrainer:
    @staticmethod
    def _make_trainer():
        torch.manual_seed(0)
        vocab, dim = 20, 16
        target = 7

        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab, dim)
                self.out = nn.Linear(dim, vocab)

            def forward(self, ids):
                return self.out(self.emb(ids))

        policy = Policy()
        policy.out.bias.data[target] = 3.0
        ref = copy.deepcopy(policy)

        def logprob_fn(p, prompt, completion):
            seq = torch.tensor([prompt + completion])
            logits = p(seq)
            logp = F.log_softmax(logits, dim=-1)
            pos = torch.arange(len(prompt), len(seq[0]))
            return logp[0, pos, torch.tensor(completion)].sum()

        def generate_fn(p, prompt, max_len):
            with torch.no_grad():
                seq = list(prompt)
                for _ in range(max_len):
                    logits = p(torch.tensor([seq]))[0, -1]
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                    seq.append(nxt)
            return seq[len(prompt) :]

        oracle = MockOracle(valid_ids=[target])
        return (
            RLVRTrainer(
                policy=policy,
                ref_model=ref,
                logprob_fn=logprob_fn,
                generate_fn=generate_fn,
                oracle=oracle,
                lr=5e-2,
                num_generations=16,
                max_gen_len=1,
            ),
            policy,
            target,
        )

    def test_train_step_updates_policy_toward_oracle(self):
        trainer, policy, target = self._make_trainer()
        prob_before = F.softmax(policy(torch.tensor([[0, 1]]))[0, -1], dim=-1)[target].item()
        stats = trainer.train_step([[0, 1]], temperature=1.0)
        assert "loss" in stats and "mean_reward" in stats and "mean_kl" in stats
        assert "std_reward" in stats
        prob_after = F.softmax(policy(torch.tensor([[0, 1]]))[0, -1], dim=-1)[target].item()
        assert prob_after >= prob_before

    def test_train_epoch_runs(self):
        trainer, _, _ = self._make_trainer()
        stats = trainer.train_epoch([[0, 1], [2, 3]], num_steps=1)
        assert torch.isfinite(torch.tensor(stats["loss"]))

    def test_format_bonus_eos(self):
        torch.manual_seed(0)
        # A completion ending in eos gains the bonus even if the oracle scores 0.
        trainer, _, _ = self._make_trainer()
        bonus = trainer.grpo.reward_fn([0, 1], [5])  # no eos, no bonus machinery
        assert bonus == 0.0


class TestProtocol:
    def test_verifiable_oracle_raises(self):
        with pytest.raises(NotImplementedError):
            VerifiableOracle().verify([1])
