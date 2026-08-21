"""
cadgenesis.evaluation.cad_bench
===============================
CAD-specific evaluation benchmark: does the model actually *make valid CAD*?

General-purpose metrics (perplexity, token accuracy) do not measure what a
CAD-generation model is for.  This suite reports:

* ``compile_rate``         — fraction of generations accepted by the oracle
                             (geometry valid / executable).
* ``constraint_sat_rate``  — fraction satisfying an optional constraint check.
* ``exact_match``          — fraction identical to the reference program.
* ``sequence_accuracy``    — mean token-level match rate vs the reference.
* ``mean_confidence``      — the model's own calibrated confidence signal.
* ``oracle_avg_reward``    — mean continuous oracle reward over generations.

The oracle is any :class:`~cadgenesis.distillation.rlvr.VerifiableOracle`
(the :class:`DesignOracle` adapter around the CAD execution engine is the
default), so the same verifier powers RLVR training, test-time search, and
evaluation — a closed loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cadgenesis.distillation.rlvr import VerifiableOracle
from cadgenesis.evaluation.report_generator import ReportGenerator
from cadgenesis.inference.engine import CADInferenceEngine


class ProgramOracle(VerifiableOracle):
    """Oracle over the real DSL: decode ids -> tokens, run the native
    geometry validator (no FreeCAD required).  Valid programs get 0.7,
    invalid get 0.0 — a signal random/frequency baselines cannot game."""

    def __init__(self, tokenizer, validator=None):
        self.tokenizer = tokenizer
        if validator is None:
            from cadgenesis.execution.geometry_validation import validate_program

            validator = validate_program
        self.validator = validator

    def verify(self, completion_ids: list[int]) -> float:
        tokens = self.tokenizer.decode_cad_sequence(list(completion_ids))
        return 0.7 if self.validator(tokens) else 0.0


@dataclass
class CADBenchItem:
    """One benchmark problem: a prompt + an optional reference program."""

    prompt: str
    reference_ids: list[int] | None = None


@dataclass
class CADBenchResult:
    """Aggregate metrics over a benchmark run."""

    compile_rate: float
    constraint_sat_rate: float
    exact_match: float
    sequence_accuracy: float
    mean_confidence: float
    oracle_avg_reward: float
    num_samples: int

    def summary(self) -> str:
        return (
            f"compile_rate={self.compile_rate:.3f} "
            f"constraint_sat={self.constraint_sat_rate:.3f} "
            f"exact_match={self.exact_match:.3f} "
            f"seq_acc={self.sequence_accuracy:.3f} "
            f"conf={self.mean_confidence:.3f} "
            f"reward={self.oracle_avg_reward:.3f} "
            f"(n={self.num_samples})"
        )


class CADBenchmark:
    """Run greedy + sampled generations over a dataset and score them."""

    def __init__(
        self,
        items: list[CADBenchItem],
        oracle: VerifiableOracle | None = None,
        constraint_checker=None,
    ):
        if not items:
            raise ValueError("CADBenchmark requires at least one item.")
        self.items = items
        self.oracle = oracle or DesignOracle()
        self.constraint_checker = constraint_checker  # callable(ids) -> bool

    def evaluate(
        self,
        engine: CADInferenceEngine,
        max_len: int = 64,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> CADBenchResult:
        """Score the model over the item set (greedy/sampled generation)."""

        def sample(item: CADBenchItem) -> tuple[list[int], float]:
            result = engine.sample(
                item.prompt,
                max_len=max_len,
                temperature=temperature,
                use_cache=use_cache,
            )
            return result.ids, float(result.confidence)

        return self._score(sample, max_len=max_len)

    def evaluate_baseline(
        self,
        baseline: "Baseline",
        max_len: int = 64,
    ) -> CADBenchResult:
        """Score a baseline policy over the item set (no model required)."""

        def sample(item: CADBenchItem) -> tuple[list[int], float]:
            return baseline.sample(item.prompt, max_len=max_len), 0.0

        return self._score(sample, max_len=max_len)

    def _score(
        self,
        ids_fn: Callable[[CADBenchItem], tuple[list[int], float]],
        max_len: int = 64,
    ) -> CADBenchResult:
        compiles = 0
        sat = 0
        exact = 0
        acc_total = 0.0
        reward_total = 0.0
        conf_total = 0.0
        n = 0

        for item in self.items:
            ids, confidence = ids_fn(item)
            n += 1
            reward = float(self.oracle.verify(ids))
            reward_total += reward
            if reward >= 0.7:
                compiles += 1
            if self.constraint_checker is not None and self.constraint_checker(ids):
                sat += 1
            if item.reference_ids is not None:
                if tuple(ids) == tuple(item.reference_ids):
                    exact += 1
                acc_total += self._sequence_accuracy(ids, item.reference_ids)
            conf_total += confidence

        denom = max(n, 1)
        has_ref = any(item.reference_ids is not None for item in self.items)
        return CADBenchResult(
            compile_rate=compiles / denom,
            constraint_sat_rate=(sat / denom if self.constraint_checker else 0.0),
            exact_match=(exact / denom if has_ref else 0.0),
            sequence_accuracy=(acc_total / n if has_ref else 0.0),
            mean_confidence=conf_total / denom,
            oracle_avg_reward=reward_total / denom,
            num_samples=n,
        )

    @staticmethod
    def _sequence_accuracy(pred: list[int], ref: list[int]) -> float:
        if not ref:
            return 0.0
        matches = sum(1 for a, b in zip(pred, ref, strict=False) if a == b)
        return matches / max(len(ref), 1)


class Baseline:
    """Base class for non-model baselines: sample ids for a prompt."""

    def sample(self, prompt: str, max_len: int = 64) -> list[int]:
        raise NotImplementedError


class RandomBaseline(Baseline):
    """Uniform-random token ids (no CAD knowledge)."""

    def __init__(self, vocab_size: int, seed: int = 0, min_len: int = 4):
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.rng = random.Random(seed)

    def sample(self, prompt: str, max_len: int = 64) -> list[int]:
        length = self.rng.randint(self.min_len, max(max_len, self.min_len))
        return [self.rng.randrange(self.vocab_size) for _ in range(length)]


class FrequencyBaseline(Baseline):
    """Repeats the most common token ids from a corpus (n-gram-ish prior).

    ``token_ids`` must be ordered most-frequent first; the baseline emits a
    cycle of the top ``k`` ids, giving a deterministic, CAD-agnostic prior
    that a trained model must beat.
    """

    def __init__(self, token_ids: list[int], top_k: int = 8):
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        self.token_ids = token_ids
        self.top_k = max(1, min(top_k, len(token_ids)))

    def sample(self, prompt: str, max_len: int = 64) -> list[int]:
        length = max_len if max_len > 0 else 8
        return [self.token_ids[i % self.top_k] for i in range(length)]


def write_benchmark_report(
    path: str,
    results: list[tuple[str, CADBenchResult]],
    title: str = "CAD Benchmark Report",
) -> str:
    """Write a markdown benchmark report (per-entry table + aggregate row).

    ``results`` is a list of ``(label, CADBenchResult)`` pairs, e.g.
    ``[("model", model_result), ("random", random_result)]``.
    Returns the rendered markdown text.
    """
    rows = [
        {
            "entry": label,
            "compile_rate": result.compile_rate,
            "constraint_sat_rate": result.constraint_sat_rate,
            "exact_match": result.exact_match,
            "sequence_accuracy": result.sequence_accuracy,
            "mean_confidence": result.mean_confidence,
            "oracle_avg_reward": result.oracle_avg_reward,
            "num_samples": result.num_samples,
        }
        for label, result in results
    ]
    sections: dict[str, Any] = {"Benchmark Entries": rows}
    if rows:
        best = max(rows, key=lambda r: r["oracle_avg_reward"])
        sections["Aggregate"] = {
            "best_entry": best["entry"],
            "best_oracle_avg_reward": best["oracle_avg_reward"],
            "entries": len(rows),
        }
    text = ReportGenerator().render_markdown(sections, title=title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text


__all__ = [
    "Baseline",
    "CADBenchItem",
    "CADBenchResult",
    "CADBenchmark",
    "FrequencyBaseline",
    "ProgramOracle",
    "RandomBaseline",
    "write_benchmark_report",
]
