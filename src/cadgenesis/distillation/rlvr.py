"""
cadgenesis.distillation.rlvr
============================
RLVR — reinforcement learning with *verifiable* rewards.

The highest-leverage post-training method for a CAD-generation model: instead of
learning from human preferences, the reward comes from an automatic *oracle*
that checks whether the generated program is actually correct — the CAD
compiles, the geometry validates, the constraints hold.  This turns GRPO into a
self-improving loop with no human labels (DeepSeek-R1-style RLVR).

This module provides:

* :class:`DesignOracle` — verifiable reward from the CAD execution engine
  (geometry validity + manufacturability).
* :class:`MockOracle` — deterministic oracle for tests / sanity checks.
* :class:`RLVRTrainer` — GRPO trainer whose reward is computed by an oracle;
  shares the exact GRPO surrogate so it inherits group-relative advantages and
  KL regularisation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from cadgenesis.distillation.grpo import GRPOTrainer


class VerifiableOracle:
    """Protocol: maps a completion (list of CAD token ids) to a float reward."""

    def verify(self, completion_ids: list[int]) -> float:
        raise NotImplementedError


class DesignOracle(VerifiableOracle):
    """
    Verifiable reward from the CAD execution/validation pipeline.

    ``completion_to_design`` translates generated token ids into a part
    descriptor dict consumed by :meth:`CADExecutionEngine.execute`; a return of
    ``None`` means "unparseable" → reward 0.  Rewards are in ``[0, 1]``:

    * ``+0.7`` when the geometry validates,
    * ``+0.3`` when it is also manufacturable.
    """

    def __init__(
        self,
        execution_engine=None,
        completion_to_design: Callable[[list[int]], dict | None] | None = None,
    ):
        if execution_engine is None:
            from cadgenesis.execution.execution_engine import CADExecutionEngine

            execution_engine = CADExecutionEngine()
        self.engine = execution_engine
        self.completion_to_design = completion_to_design or _default_design

    def verify(self, completion_ids: list[int]) -> float:
        design = self.completion_to_design(completion_ids)
        if not design:
            return 0.0
        try:
            result = self.engine.execute(
                design=design, validate=True, simulate=False, optimize=False
            )
        except Exception:
            return 0.0
        score = 0.0
        if result.is_valid_geometry:
            score += 0.7
            if result.is_manufacturable:
                score += 0.3
        return score


def _default_design(completion_ids: list[int]) -> dict | None:
    """Best-effort default: every token id becomes a box feature (permissively
    valid) unless the completion is empty — override with your own parser."""
    if not completion_ids:
        return None
    return {
        "name": "generated_part",
        "features": [
            {"type": "box", "width_m": 0.1, "height_m": 0.1, "depth_m": 0.1} for _ in completion_ids
        ],
    }


class MockOracle(VerifiableOracle):
    """Deterministic oracle: completions matching ``valid_ids`` get 1.0."""

    def __init__(self, valid_ids: list[int] | None = None):
        self.valid_ids = valid_ids or []

    def verify(self, completion_ids: list[int]) -> float:
        return 1.0 if completion_ids == self.valid_ids else 0.0


class RLVRTrainer:
    """
    GRPO with a verifiable-reward oracle.

    The reward for each sampled completion is ``oracle.verify(completion_ids)``
    (optionally plus ``format_bonus`` when the sequence ends with the EOS id).
    All other mechanics (group-relative advantages, clipping, KL penalty) are
    exactly :class:`GRPOTrainer`'s.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        logprob_fn: Callable,
        generate_fn: Callable,
        oracle: VerifiableOracle,
        lr: float = 1e-5,
        kl_coef: float = 0.01,
        clip_epsilon: float = 0.2,
        num_generations: int = 4,
        max_gen_len: int = 64,
        device: str = "cpu",
        format_bonus: float = 0.0,
        eos_id: int | None = None,
    ):
        def reward_fn(prompt: list[int], completion: list[int]) -> float:
            reward = float(oracle.verify(completion))
            if format_bonus and eos_id is not None and completion and completion[-1] == eos_id:
                reward += format_bonus
            return reward

        self.grpo = GRPOTrainer(
            policy=policy,
            ref_model=ref_model,
            logprob_fn=logprob_fn,
            generate_fn=generate_fn,
            reward_fn=reward_fn,
            lr=lr,
            kl_coef=kl_coef,
            clip_epsilon=clip_epsilon,
            num_generations=num_generations,
            max_gen_len=max_gen_len,
            device=device,
        )
        self.oracle = oracle

    def train_step(self, prompts, temperature: float = 1.0) -> dict[str, float]:
        return self.grpo.train_step(prompts, temperature)

    def train_epoch(self, prompts, num_steps: int = 1) -> dict[str, float]:
        return self.grpo.train_epoch(prompts, num_steps)


__all__ = [
    "DesignOracle",
    "MockOracle",
    "RLVRTrainer",
    "VerifiableOracle",
]
