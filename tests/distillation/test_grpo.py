"""
tests/distillation/test_grpo.py
================================
Tests for cadgenesis.distillation.grpo (GRPO trainer).
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.distillation.grpo import GRPOTrainer, make_validity_reward

VOCAB_SIZE = 20
TARGET_TOKEN = 7


class TinyPolicy(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids))


def logprob_fn(model: nn.Module, prompt_ids: list, completion_ids: list) -> torch.Tensor:
    seq = torch.tensor([prompt_ids + completion_ids])
    logits = model(seq)
    logp = F.log_softmax(logits, dim=-1)
    positions = torch.arange(len(prompt_ids), len(seq[0]))
    return logp[0, positions, torch.tensor(completion_ids)].sum()


def generate_fn(model: nn.Module, prompt_ids: list, max_len: int) -> list:
    with torch.no_grad():
        seq = list(prompt_ids)
        for _ in range(max_len):
            logits = model(torch.tensor([seq]))[0, -1]
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            seq.append(next_id)
    return seq[len(prompt_ids) :]


def reward_fn(prompt_ids: list, completion_ids: list) -> float:
    return 1.0 if completion_ids and completion_ids[-1] == TARGET_TOKEN else 0.0


def sample_prompts() -> list:
    return [[0, 1], [2, 3], [4, 5], [6, 8]]


@pytest.fixture
def trainer() -> GRPOTrainer:
    torch.manual_seed(0)
    policy = TinyPolicy()
    policy.out.bias.data[TARGET_TOKEN] = 3.0
    ref_model = copy.deepcopy(policy)
    return GRPOTrainer(
        policy=policy,
        ref_model=ref_model,
        logprob_fn=logprob_fn,
        generate_fn=generate_fn,
        reward_fn=reward_fn,
        lr=5e-2,
        kl_coef=0.01,
        clip_epsilon=0.2,
        num_generations=32,
        max_gen_len=1,
    )


class TestTrainStep:
    def test_stats_dict(self, trainer):
        stats = trainer.train_step(sample_prompts())
        assert set(stats) == {"loss", "mean_reward", "mean_kl", "std_reward"}
        for value in stats.values():
            assert math.isfinite(value)
        assert 0.0 <= stats["mean_reward"] <= 1.0
        assert stats["std_reward"] >= 0.0


class TestLearning:
    def test_rewards_improve_over_epochs(self, trainer):
        prompts = sample_prompts()
        first = trainer.train_epoch(prompts, num_steps=1)["mean_reward"]
        last = first
        for _ in range(4):
            last = trainer.train_epoch(prompts, num_steps=1)["mean_reward"]
        assert last > first


class TestGradients:
    def test_policy_gradients_flow(self, trainer):
        trainer.train_step(sample_prompts())
        grad = trainer.policy.out.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert trainer.policy.out.bias.grad is not None


class TestValidityReward:
    def test_validity_reward_wraps_validator(self):
        reward = make_validity_reward(lambda completion_ids: completion_ids[-1] == TARGET_TOKEN)
        assert reward([0, 1], [3, TARGET_TOKEN]) == 1.0
        assert reward([0, 1], [3, 4]) == 0.0
