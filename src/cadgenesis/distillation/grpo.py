"""
cadgenesis.distillation.grpo
============================
GRPO (Group Relative Policy Optimization, DeepSeek-R1 style) trainer.

GRPO samples ``num_generations`` completions per prompt, scores each with a
reward function, and normalizes rewards into group-relative advantages::

    adv_i = (r_i - mean(r_group)) / (std(r_group) + 1e-8)

The policy is updated with a REINFORCE-style policy-gradient surrogate on
those advantages plus an additive KL penalty against a frozen reference model::

    loss = -mean(adv * p_logp) + kl_coef * mean((p_logp - r_logp).detach() * p_logp)

where ``p_logp`` is the differentiable log-probability of the completion under
the policy and ``r_logp`` the (detached) log-probability under the reference
model.  Group-relative advantage normalization is the GRPO mechanism; unlike
clipped-ratio GRPO, the surrogate is linear in ``p_logp``, so no old-policy
snapshot (``copy.deepcopy`` of the policy) is required and the importance-ratio
clipping of PPO/GRPO is omitted.

Differentiability: ``logprob_fn`` MUST return a differentiable ``torch.Tensor``
with ``requires_grad``; callers compute log-probs directly from raw model
logits (e.g. ``log_softmax(model(prompt + completion))`` gathered at the
completion positions), which makes the surrogate and the KL term
back-propagate into the policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

__all__ = ["GRPOTrainer", "make_validity_reward"]


class GRPOTrainer:
    """
    Trains a policy model with GRPO on a reward function.

    ``logprob_fn(model, prompt_ids, completion_ids)`` returns a differentiable
    scalar ``torch.Tensor`` with the log-probability of the completion given
    the prompt; it is evaluated on the policy and on the frozen reference
    model.  ``generate_fn(model, prompt_ids, max_len)`` samples one completion
    (temperature sampling is the caller's responsibility) and ``reward_fn(
    prompt_ids, completion_ids)`` returns a scalar reward.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        logprob_fn: Callable,
        generate_fn: Callable,
        reward_fn: Callable,
        lr: float = 1e-5,
        kl_coef: float = 0.01,
        clip_epsilon: float = 0.2,
        num_generations: int = 4,
        max_gen_len: int = 64,
        device: str = "cpu",
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.logprob_fn = logprob_fn
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.num_generations = num_generations
        self.max_gen_len = max_gen_len
        self.device = device
        self.policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    def train_step(self, prompts: list[list[int]], temperature: float = 1.0) -> dict[str, float]:
        """
        One GRPO update over the given prompts.

        Samples ``num_generations`` completions per prompt, computes
        group-relative advantages, and applies the REINFORCE-style surrogate
        plus the KL penalty.  ``temperature`` is accepted for API compatibility;
        the caller's ``generate_fn`` is responsible for temperature sampling.
        """
        self.ref_model.eval()

        p_logp_list: list[torch.Tensor] = []
        r_logp_list: list[torch.Tensor] = []
        rewards_list: list[float] = []
        advantages_list: list[float] = []

        for prompt in prompts:
            with torch.no_grad():
                completions = [
                    self.generate_fn(self.policy, prompt, self.max_gen_len)
                    for _ in range(self.num_generations)
                ]
            group_rewards = [self.reward_fn(prompt, completion) for completion in completions]
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards))
            for completion, reward in zip(completions, group_rewards, strict=False):
                p_logp = self.logprob_fn(self.policy, prompt, completion)
                with torch.no_grad():
                    r_logp = self.logprob_fn(self.ref_model, prompt, completion)
                p_logp_list.append(p_logp)
                r_logp_list.append(r_logp)
                rewards_list.append(reward)
                advantages_list.append((reward - mean_r) / (std_r + 1e-8))

        p_logp = torch.stack(p_logp_list)
        r_logp = torch.stack(r_logp_list).detach()
        advantages = torch.tensor(advantages_list, dtype=p_logp.dtype, device=self.device)

        kl = (p_logp - r_logp).detach() * p_logp
        loss = -torch.mean(advantages * p_logp) + self.kl_coef * torch.mean(kl)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        mean_reward = sum(rewards_list) / len(rewards_list)
        std_reward = math.sqrt(
            sum((r - mean_reward) ** 2 for r in rewards_list) / len(rewards_list)
        )
        return {
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "mean_kl": torch.mean(p_logp - r_logp).item(),
            "std_reward": std_reward,
        }

    def train_epoch(self, prompts: list[list[int]], num_steps: int = 1) -> dict[str, float]:
        """
        Runs ``train_step`` ``num_steps`` times and averages the returned stats.
        """
        stats = [self.train_step(prompts) for _ in range(num_steps)]
        return {key: sum(step[key] for step in stats) / len(stats) for key in stats[0]}


def make_validity_reward(
    validator: Callable[[list[int]], bool],
) -> Callable[[list[int], list[int]], float]:
    """
    Wraps a validity check ``validator(completion_ids) -> bool`` into a reward
    of 1.0 for valid completions and 0.0 otherwise.
    """

    def reward(prompt_ids: list[int], completion_ids: list[int]) -> float:
        return 1.0 if validator(completion_ids) else 0.0

    return reward
