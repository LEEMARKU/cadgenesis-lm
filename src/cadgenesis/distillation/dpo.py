"""cadgenesis.distillation.dpo
============================
Direct Preference Optimization (DPO) trainer.

DPO (Rafailov et al., 2023) fine-tunes a policy against a frozen
reference model using an implicit reward derived from the log-probability
ratio of chosen and rejected completions:

    r_implicit(x, y) = beta * (log pi_policy(y|x) - log pi_ref(y|x))

The reference model provides regularization: it anchors the policy to the
reference distribution so the policy cannot drift arbitrarily far.  The
temperature `beta` controls the strength of that anchoring -- a larger
beta penalizes deviation from the reference more strongly, while a
smaller beta allows the policy to move further.  The implicit reward is
never modeled explicitly; instead the policy is trained directly with the
binary logistic loss

    L_DPO = -log_sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

which prefers completions whose implicit reward exceeds that of their
rejected counterparts, optionally with label smoothing to soften the
preference signal.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DPOTrainer"]


class DPOTrainer:
    """Train a policy against a fixed reference model with DPO."""

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        logprob_fn: Callable[[nn.Module, list[int], list[int]], torch.Tensor],
        lr: float = 1e-5,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.logprob_fn = logprob_fn
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.device = device
        self.policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    def dpo_loss(
        self,
        prompt_ids: list[int],
        chosen_ids: list[int],
        rejected_ids: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        p_chosen = self.logprob_fn(self.policy, prompt_ids, chosen_ids)
        p_rejected = self.logprob_fn(self.policy, prompt_ids, rejected_ids)
        with torch.no_grad():
            r_chosen = self.logprob_fn(self.ref_model, prompt_ids, chosen_ids)
            r_rejected = self.logprob_fn(self.ref_model, prompt_ids, rejected_ids)
        log_ratio_chosen = (p_chosen - r_chosen) * self.beta
        log_ratio_rejected = (p_rejected - r_rejected) * self.beta
        diff = log_ratio_chosen - log_ratio_rejected
        if self.label_smoothing:
            loss = -(1.0 - self.label_smoothing) * F.logsigmoid(
                diff
            ) - self.label_smoothing * F.logsigmoid(-diff)
        else:
            loss = -F.logsigmoid(diff)
        return (
            loss,
            {
                "dpo": float(loss.detach().item()),
                "chosen_margin": float(log_ratio_chosen.detach().item()),
                "rejected_margin": float(log_ratio_rejected.detach().item()),
            },
        )

    def train_step(
        self,
        prompts: list[list[int]],
        chosen: list[list[int]],
        rejected: list[list[int]],
    ) -> dict[str, float]:
        if not (len(prompts) == len(chosen) == len(rejected)):
            raise ValueError("prompts, chosen, and rejected must contain the same number of pairs")
        if len(prompts) == 0:
            raise ValueError("cannot train on an empty batch")
        losses = []
        margins = []
        for prompt_ids, chosen_ids, rejected_ids in zip(prompts, chosen, rejected, strict=False):
            loss, stats = self.dpo_loss(prompt_ids, chosen_ids, rejected_ids)
            losses.append(loss)
            margins.append(stats["chosen_margin"])
        batch_loss = torch.stack(losses).mean()
        self.policy_optimizer.zero_grad()
        batch_loss.backward()
        self.policy_optimizer.step()
        return {
            "loss": float(batch_loss.detach().item()),
            "chosen_margin": float(sum(margins) / len(margins)),
        }

    def train_epoch(
        self,
        prompts: list[list[int]],
        chosen: list[list[int]],
        rejected: list[list[int]],
        num_steps: int = 1,
    ) -> dict[str, float]:
        losses = []
        margins = []
        for _ in range(num_steps):
            stats = self.train_step(prompts, chosen, rejected)
            losses.append(stats["loss"])
            margins.append(stats["chosen_margin"])
        return {
            "loss": float(sum(losses) / len(losses)),
            "chosen_margin": float(sum(margins) / len(margins)),
        }
