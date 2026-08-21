"""
cadgenesis.training.rlvr_pipeline
=================================
Closed-loop orchestration for the 2025-2026 post-training stack.

Binds the frontier pieces into one surface:

* **RLVR** — GRPO post-training whose reward comes from a *verifiable* CAD
  oracle (:class:`~cadgenesis.distillation.rlvr.DesignOracle`).
* **CAD evaluation** — the same oracle (and :class:`CADBenchmark`) measures
  whether generations actually compile / validate.
* **Test-time compute** — best-of-n, self-consistency and MCTS search over the
  trained model, again oracle-driven.
* **EAGLE** — a learned draft head for greedy-preserving speculative decoding.

The verifier is shared end-to-end, so training, evaluation and inference all
answer the same question: "did the model generate valid CAD?"  — a single
closed loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.distillation.rlvr import DesignOracle, RLVRTrainer, VerifiableOracle
from cadgenesis.evaluation.cad_bench import CADBenchItem, CADBenchmark, CADBenchResult
from cadgenesis.inference.engine import CADInferenceEngine, GenerationResult
from cadgenesis.inference.mcts import best_of_n, mcts, self_consistency

if TYPE_CHECKING:
    from cadgenesis.inference.eagle import EagleDraftHead


def _type_id_of(tokenizer, token_id: int) -> int:
    try:
        return tokenizer.vocab.type_id_of(int(token_id))
    except KeyError:
        return 0


class RLVRPipeline:
    """
    One object for the RLVR + verifiable-eval + test-time-compute loop.

    Parameters
    ----------
    model : nn.Module
        The policy to post-train.
    tokenizer : AutonomousCADTokenizer
        Tokenizer used to encode prompts / type-tag completions.
    oracle : VerifiableOracle, optional
        Reward / verification oracle.  Defaults to :class:`DesignOracle`
        (geometry validity + manufacturability through the execution engine).
    device : str
        Compute device for the policy forward passes.
    completion_to_design : Callable, optional
        Parser ``list[token_id] -> design dict`` for :class:`DesignOracle`.
    lr, kl_coef, clip_epsilon, num_generations, max_gen_len
        RLVR / GRPO hyperparameters (see :class:`RLVRTrainer`).
    format_bonus : float
        Extra reward for completions terminated by EOS.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        oracle: VerifiableOracle | None = None,
        device: str = "cpu",
        completion_to_design: Callable[[list[int]], dict | None] | None = None,
        lr: float = 1e-5,
        kl_coef: float = 0.01,
        clip_epsilon: float = 0.2,
        num_generations: int = 4,
        max_gen_len: int = 64,
        format_bonus: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_gen_len = max_gen_len
        self.oracle = oracle or DesignOracle(completion_to_design=completion_to_design)
        self.engine = CADInferenceEngine(model, tokenizer, device=device)
        self.format_bonus = format_bonus
        self.eos_id = tokenizer.eos_id

        self.trainer = RLVRTrainer(
            policy=model,
            ref_model=model,
            logprob_fn=self._logprob_fn,
            generate_fn=self._generate_fn,
            oracle=self.oracle,
            lr=lr,
            kl_coef=kl_coef,
            clip_epsilon=clip_epsilon,
            num_generations=num_generations,
            max_gen_len=max_gen_len,
            device=device,
            format_bonus=format_bonus,
            eos_id=self.eos_id,
        )

    # ------------------------------------------------------------------ RLVR

    def train(
        self,
        prompts: list[str],
        steps: int = 1,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Encode the natural-language prompts and run ``steps`` GRPO updates."""
        encoded = [self.tokenizer.encode_text(p) for p in prompts]
        stats: dict[str, float] = {}
        for _ in range(steps):
            step_stats = self.trainer.train_step(encoded, temperature)
            for key, value in step_stats.items():
                stats[key] = stats.get(key, 0.0) + value / steps
        return stats

    def _logprob_fn(
        self, policy: nn.Module, prompt: list[int], completion: list[int]
    ) -> torch.Tensor:
        """Sum of log p(token | prefix) over the completion (excl. its head)."""
        if not completion:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        src = torch.tensor([prompt])
        tgt = torch.tensor([completion])
        types = torch.tensor([[_type_id_of(self.tokenizer, i) for i in completion]])
        logits, _ = policy(src, tgt, types)
        logp = F.log_softmax(logits[0, :-1], dim=-1)
        targets = torch.tensor(completion[1:], dtype=torch.long)
        if targets.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return logp[torch.arange(targets.numel()), targets].sum()

    def _generate_fn(self, policy: nn.Module, prompt: list[int], max_len: int) -> list[int]:
        """Sample a CAD completion (content ids, BOS stripped) for a prompt."""
        text = self.tokenizer.decode_text(prompt)
        return self.engine.sample(text, max_len=max_len, temperature=1.0).ids

    # ------------------------------------------------------------- evaluation

    def evaluate(
        self,
        prompts: list[str] | None = None,
        items: list[CADBenchItem] | None = None,
        max_len: int | None = None,
        temperature: float = 1.0,
        use_cache: bool = True,
        constraint_checker=None,
    ) -> CADBenchResult:
        """
        Score the policy on a benchmark.  Either pass ``items`` explicitly or
        derive one ``CADBenchItem`` per prompt.
        """
        max_len = max_len or self.max_gen_len
        if items is None:
            items = [CADBenchItem(prompt=p) for p in (prompts or [])]
        bench = CADBenchmark(items, oracle=self.oracle, constraint_checker=constraint_checker)
        return bench.evaluate(
            self.engine,
            max_len=max_len,
            temperature=temperature,
            use_cache=use_cache,
        )

    # ------------------------------------------------------- test-time compute

    def best_of_n(
        self,
        text: str,
        n: int = 8,
        max_len: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[GenerationResult, float]:
        return best_of_n(
            self.engine,
            text,
            self.oracle,
            n=n,
            max_len=max_len or self.max_gen_len,
            temperature=temperature,
        )

    def self_consistency(
        self,
        text: str,
        n: int = 8,
        max_len: int | None = None,
        temperature: float = 1.0,
    ) -> GenerationResult:
        return self_consistency(
            self.engine,
            text,
            n=n,
            max_len=max_len or self.max_gen_len,
            temperature=temperature,
        )

    def mcts(
        self,
        text: str,
        iterations: int = 8,
        max_len: int | None = None,
        temperature: float = 1.0,
        branch: int = 3,
        rollout_len: int = 4,
        c: float = 1.4,
    ) -> tuple[GenerationResult, float]:
        return mcts(
            self.engine,
            text,
            self.oracle,
            iterations=iterations,
            max_len=max_len or self.max_gen_len,
            temperature=temperature,
            branch=branch,
            rollout_len=rollout_len,
            c=c,
        )

    # ------------------------------------------------------------------- EAGLE

    def train_eagle(
        self,
        sequences: list[list[int]],
        steps: int = 60,
        lr: float = 1e-3,
        num_heads: int = 4,
    ) -> nn.Module:
        """
        Collect ``(hidden, next_token)`` pairs from the policy over ``sequences``
        and fine-tune a fresh :class:`EagleDraftHead`.  Returns the head.
        """
        from cadgenesis.inference.eagle import (
            EagleDraftHead,
            collect_hidden_pairs,
            train_eagle,
        )

        head = EagleDraftHead(
            d_model=cast(int, self.model.d_model),
            num_heads=num_heads,
            vocab_size=cast(int, self.model.cad_vocab_size),
        )
        pairs = collect_hidden_pairs(
            self.model, sequences, lambda i: _type_id_of(self.tokenizer, i)
        )
        train_eagle(head, self.model, pairs, steps=steps, lr=lr, device=self.device)
        return head

    def speculative(
        self,
        text: str,
        draft_head: EagleDraftHead,
        max_len: int = 64,
        num_speculative_tokens: int = 4,
    ) -> GenerationResult:
        """Greedy-preserving EAGLE speculative decoding (falls back to greedy)."""
        from cadgenesis.inference.eagle import speculative_eagle

        return speculative_eagle(
            self.engine,
            text,
            draft_head,
            max_len=max_len,
            num_speculative_tokens=num_speculative_tokens,
            device=self.device,
        )

    # ------------------------------------------------------------------ build

    @classmethod
    def from_config(
        cls,
        config,
        tokenizer,
        device: str = "cpu",
        **kwargs,
    ) -> RLVRPipeline:
        """Build the policy from a :class:`CADConfig` and wrap it."""
        from cadgenesis.transformer.geometry_transformer import (
            GeometryAwareTransformer,
        )

        model = GeometryAwareTransformer(config)
        return cls(model, tokenizer, device=device, **kwargs)


__all__ = ["RLVRPipeline"]
