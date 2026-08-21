"""
cadgenesis.transformer.self_designing.architecture
=============================================
Architecture search space, concrete architecture specifications, and the
Neural Architecture Search (NAS) controller for CADGenesis-LM v2.0.

An ``ArchitectureSpec`` is a complete, validated description of a transformer
architecture (depth, width, head layout, MoE switch).  It can be materialised
into a ``ModelConfig`` for the existing ``GeometryAwareTransformer`` backbone
— nothing in the backbone is rebuilt.

``NeuralArchitectureSearch`` explores the search space and returns the best
specification measured by an ``ArchitectureEvaluator``.  Two strategies are
provided:

* ``random_search``  — sample ``iterations`` specs, keep the best.
* ``evolutionary``   — maintain a small population, mutate the best member
  each generation (µ+λ style).

Complexity
----------
    Random search:   O(I · E)   where I = iterations, E = eval cost
    Evolutionary:    O(G · P · E) where G = generations, P = population size
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import cast

from cadgenesis.config import CADConfig, ModelConfig

# ---------------------------------------------------------------------------
# ArchitectureSpec
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureSpec:
    """
    A validated description of a candidate transformer architecture.

    The attention head tuple must sum to ``nhead`` and ``nhead`` must divide
    ``d_model``.  ``use_moe`` swaps every block's SwiGLU FFN for a sparse MoE.
    """

    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1

    self_attn_heads: int = 2
    geometry_attn_heads: int = 1
    constraint_attn_heads: int = 0
    memory_attn_heads: int = 1
    agent_attn_heads: int = 0
    uncertainty_attn_heads: int = 0

    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2

    # Experimental subsystem switches (lean default = OFF).  Kept in the spec
    # so search/evolution candidates that activate agent/memory heads also
    # enable the subsystem that produces those head inputs.
    use_multi_agent_system: bool = False
    use_memory_system: bool = False
    use_neuro_symbolic_reasoning: bool = False
    use_rlaf_reward_model: bool = False
    use_confidence_head: bool = True

    def validate(self) -> ArchitectureSpec:
        total_heads = (
            self.self_attn_heads
            + self.geometry_attn_heads
            + self.constraint_attn_heads
            + self.memory_attn_heads
            + self.agent_attn_heads
            + self.uncertainty_attn_heads
        )
        if total_heads != self.nhead:
            raise ValueError(f"Head counts sum to {total_heads} but nhead={self.nhead}.")
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}.")
        if self.use_moe and (
            self.num_experts < 1 or not (1 <= self.top_k_experts <= self.num_experts)
        ):
            raise ValueError("MoE requires 1 <= top_k_experts <= num_experts.")
        return self

    def to_model_config(self) -> ModelConfig:
        """Materialise into a ModelConfig for the existing backbone."""
        return ModelConfig(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            self_attn_heads=self.self_attn_heads,
            geometry_attn_heads=self.geometry_attn_heads,
            constraint_attn_heads=self.constraint_attn_heads,
            memory_attn_heads=self.memory_attn_heads,
            agent_attn_heads=self.agent_attn_heads,
            uncertainty_attn_heads=self.uncertainty_attn_heads,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            use_multi_agent_system=self.use_multi_agent_system or self.agent_attn_heads > 0,
            use_memory_system=self.use_memory_system or self.memory_attn_heads > 0,
            use_neuro_symbolic_reasoning=self.use_neuro_symbolic_reasoning,
            use_rlaf_reward_model=self.use_rlaf_reward_model,
            use_confidence_head=self.use_confidence_head,
        )

    def to_config(self, base: CADConfig | None = None) -> CADConfig:
        """Build a full CADConfig (tokenizer/training inherited from ``base``)."""
        cfg = base or CADConfig.mini()
        cfg.model = self.to_model_config()
        return cfg

    def signature(self) -> str:
        """Stable short hash for checkpoint / experiment bookkeeping."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_model_config(cls, m: ModelConfig) -> ArchitectureSpec:
        return cls(
            num_encoder_layers=m.num_encoder_layers,
            num_decoder_layers=m.num_decoder_layers,
            d_model=m.d_model,
            nhead=m.nhead,
            dim_feedforward=m.dim_feedforward,
            dropout=m.dropout,
            self_attn_heads=m.self_attn_heads,
            geometry_attn_heads=m.geometry_attn_heads,
            constraint_attn_heads=m.constraint_attn_heads,
            memory_attn_heads=m.memory_attn_heads,
            agent_attn_heads=m.agent_attn_heads,
            uncertainty_attn_heads=m.uncertainty_attn_heads,
            use_moe=m.use_moe,
            num_experts=m.num_experts,
            top_k_experts=m.top_k_experts,
            use_multi_agent_system=m.use_multi_agent_system,
            use_memory_system=m.use_memory_system,
            use_neuro_symbolic_reasoning=m.use_neuro_symbolic_reasoning,
            use_rlaf_reward_model=m.use_rlaf_reward_model,
            use_confidence_head=m.use_confidence_head,
        ).validate()

    def __post_init__(self):
        self.validate()

    def __repr__(self) -> str:
        return (
            f"ArchitectureSpec(E={self.num_encoder_layers},D={self.num_decoder_layers},"
            f"d={self.d_model},h={self.nhead},ffn={self.dim_feedforward},"
            f"heads=({self.self_attn_heads},{self.geometry_attn_heads},"
            f"{self.constraint_attn_heads},{self.memory_attn_heads},"
            f"{self.agent_attn_heads},{self.uncertainty_attn_heads}),"
            f"moe={self.use_moe})"
        )


# ---------------------------------------------------------------------------
# ArchitectureSearchSpace
# ---------------------------------------------------------------------------

# Head layouts that sum to nhead=4 (used by search space sampling).
_HEAD_LAYOUTS_4: list[tuple[int, int, int, int, int, int]] = [
    (2, 1, 0, 1, 0, 0),
    (1, 1, 0, 1, 1, 0),
    (2, 0, 1, 1, 0, 0),
    (1, 1, 0, 0, 1, 1),
    (2, 1, 1, 0, 0, 0),
]

# Head layouts that sum to nhead=8.
_HEAD_LAYOUTS_8: list[tuple[int, int, int, int, int, int]] = [
    (2, 2, 1, 1, 1, 1),
    (2, 2, 1, 2, 1, 0),
    (3, 1, 1, 1, 1, 1),
    (2, 3, 1, 1, 0, 1),
    (3, 2, 1, 0, 1, 1),
]


@dataclass
class ArchitectureSearchSpace:
    """
    Compact, test-friendly search space.  All sampled specs are validated.

    Fields are tuples of allowed values; ``sample(rng)`` picks uniformly.
    """

    layer_choices: tuple[int, ...] = (1, 2, 3, 4)
    d_model_choices: tuple[int, ...] = (64, 128)
    ffn_ratio_choices: tuple[float, ...] = (1.5, 2.0, 3.0)
    head_layouts: tuple[tuple[int, ...], ...] = field(
        default_factory=lambda: tuple(tuple(h) for h in _HEAD_LAYOUTS_4)
    )
    moe_choices: tuple[bool, ...] = (False, True)
    num_experts_choices: tuple[int, ...] = (2, 4)

    def sample(self, rng: random.Random | None = None) -> ArchitectureSpec:
        rng = rng or cast(random.Random, random)
        layout = list(rng.choice(self.head_layouts))
        d_model = rng.choice(self.d_model_choices)
        nhead = sum(layout)
        enc = rng.choice(self.layer_choices)
        dec = rng.choice(self.layer_choices)
        ffn = int(round(rng.choice(self.ffn_ratio_choices) * d_model / 16) * 16)
        use_moe = rng.choice(self.moe_choices)
        num_experts = rng.choice(self.num_experts_choices)
        return ArchitectureSpec(
            num_encoder_layers=enc,
            num_decoder_layers=dec,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(ffn, 64),
            dropout=0.1,
            self_attn_heads=layout[0],
            geometry_attn_heads=layout[1],
            constraint_attn_heads=layout[2],
            memory_attn_heads=layout[3],
            agent_attn_heads=layout[4],
            uncertainty_attn_heads=layout[5],
            use_moe=use_moe,
            num_experts=num_experts,
            top_k_experts=min(2, num_experts),
        )

    def mutate(self, spec: ArchitectureSpec, rng: random.Random | None = None) -> ArchitectureSpec:
        """Mutation operator for evolutionary search (returns a new spec)."""
        rng = rng or cast(random.Random, random)
        new = self.sample(rng)
        # Inherit the better-performing structural choices half the time.
        if rng.random() < 0.5:
            new.d_model = spec.d_model
        if rng.random() < 0.5:
            new.nhead = spec.nhead
            new.self_attn_heads = spec.self_attn_heads
            new.geometry_attn_heads = spec.geometry_attn_heads
            new.constraint_attn_heads = spec.constraint_attn_heads
            new.memory_attn_heads = spec.memory_attn_heads
            new.agent_attn_heads = spec.agent_attn_heads
            new.uncertainty_attn_heads = spec.uncertainty_attn_heads
        return new.validate()


# ---------------------------------------------------------------------------
# NeuralArchitectureSearch
# ---------------------------------------------------------------------------


class NeuralArchitectureSearch:
    """
    Searches the architecture space for the best-performing transformer spec.

    The evaluator is any callable ``spec -> float`` returning a *higher is
    better* quality score (see ``ArchitectureEvaluator.score``).
    """

    def __init__(
        self,
        space: ArchitectureSearchSpace | None = None,
        evaluator: Callable[[ArchitectureSpec], float] | None = None,
        seed: int = 42,
    ):
        self.space = space or ArchitectureSearchSpace()
        self.evaluator = evaluator
        self.seed = seed
        self._rng = random.Random(seed)
        self.history: list[tuple[ArchitectureSpec, float]] = []

    def _score(self, spec: ArchitectureSpec) -> float:
        if self.evaluator is None:
            raise RuntimeError("NeuralArchitectureSearch requires an evaluator.")
        return self.evaluator(spec)

    def random_search(self, iterations: int = 8) -> tuple[ArchitectureSpec, float]:
        """Uniformly sample ``iterations`` specs; return (best_spec, best_score)."""
        best_spec, best_score = None, float("-inf")
        for _ in range(iterations):
            spec = self.space.sample(self._rng)
            score = self._score(spec)
            self.history.append((spec, score))
            if best_spec is None or score > best_score:
                best_spec, best_score = spec, score
        assert best_spec is not None
        return best_spec, best_score

    def evolutionary(
        self,
        generations: int = 4,
        population_size: int = 4,
        elite_fraction: float = 0.5,
    ) -> tuple[ArchitectureSpec, float]:
        """
        µ+λ evolutionary search.  Seed the population with random samples,
        score all members, keep the best ``elite_fraction`` as parents and
        fill the rest with mutations of the elite.
        """
        population = [self.space.sample(self._rng) for _ in range(population_size)]
        scores = [self._score(s) for s in population]
        self.history.extend(zip(population, scores, strict=False))

        n_elite = max(1, int(population_size * elite_fraction))
        for _ in range(generations - 1):
            ranked = sorted(zip(population, scores, strict=False), key=lambda p: p[1], reverse=True)
            parents = [s for s, _ in ranked[:n_elite]]
            population = list(parents)
            while len(population) < population_size:
                population.append(self.space.mutate(self._rng.choice(parents), self._rng))
            scores = [self._score(s) for s in population]
            self.history.extend(zip(population, scores, strict=False))

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return population[best_idx], scores[best_idx]

    def run(
        self,
        iterations: int = 8,
        generations: int = 4,
        population_size: int = 4,
        mode: str = "random",
    ) -> tuple[ArchitectureSpec, float]:
        """Dispatch to ``random_search`` or ``evolutionary``."""
        mode = mode.lower()
        if mode == "random":
            return self.random_search(iterations)
        if mode in ("evolutionary", "evolution", "evo"):
            return self.evolutionary(generations, population_size)
        raise ValueError(f"Unknown search mode {mode!r} (choose 'random' or 'evolutionary').")

    def summary(self) -> dict:
        return {
            "history_size": len(self.history),
            "best_score": max((s for _, s in self.history), default=None),
            "best_signature": (
                self.history[self.history.index(max(self.history, key=lambda p: p[1]))][
                    0
                ].signature()
                if self.history
                else None
            ),
        }
