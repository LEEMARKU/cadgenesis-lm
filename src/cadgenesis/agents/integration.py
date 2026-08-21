"""cadgenesis.agents.integration
==============================
Pillar 5 integration adapters.

Safe, lazy adapters that bind the multi-agent platform to the rest of the
CADGenesis-LM stack: the transformer block, tokenizer, world model, memory
system, reasoning engines, execution engine, confidence engine and the
(stub-safe) continual-learning hooks.  Every adapter is optional — callers pass
in instances or let the adapter build defaults, and torch-backed pieces are
only touched when actually used.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TransformerAgentAdapter:
    """Bridges the agent platform to :class:`CADTransformerBlock`.

    Exposes the ``memory_heads`` / ``agent_heads`` the block already accepts,
    so P5 agents and P6 memory can share the same transformer.
    """

    def __init__(
        self,
        d_model: int = 1024,
        memory_heads: int = 2,
        agent_heads: int = 2,
        use_moe: bool = False,
    ) -> None:
        self.config: dict[str, Any] = {
            "d_model": d_model,
            "memory_heads": memory_heads,
            "agent_heads": agent_heads,
            "use_moe": use_moe,
        }
        self._block: Any | None = None

    @property
    def block(self) -> Any:
        """Lazily construct the transformer block."""
        if self._block is None:
            from cadgenesis.transformer import CADTransformerBlock

            self._block = CADTransformerBlock(
                d_model=self.config["d_model"],
                memory_heads=self.config["memory_heads"],
                agent_heads=self.config["agent_heads"],
                use_moe=self.config["use_moe"],
            )
        return self._block

    def heads(self) -> dict[str, int]:
        return {
            "memory_heads": self.config["memory_heads"],
            "agent_heads": self.config["agent_heads"],
        }

    def forward_batch(self, x: Any) -> Any:
        return self.block(x)


class TokenizerAdapter:
    """Bridges the platform to :class:`AutonomousCADTokenizer`."""

    def __init__(self, tokenizer: Any = None) -> None:
        self.tokenizer = tokenizer

    def _ensure(self) -> Any:
        if self.tokenizer is None:
            from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer

            self.tokenizer = AutonomousCADTokenizer.build()
        return self.tokenizer

    def encode_text(self, text: str) -> list[int]:
        return self._ensure().encode_text(text)

    def decode_text(self, tokens: list[int]) -> str:
        return self._ensure().decode_text(tokens)


class WorldModelAdapter:
    """Bridges the platform to :class:`WorldModelSystem.reason`."""

    def __init__(self, system: Any = None) -> None:
        self.system = system

    def _ensure(self) -> Any:
        if self.system is None:
            from cadgenesis.world_model import WorldModelSystem

            self.system = WorldModelSystem()
        return self.system

    def reason(self, capability: str, **kwargs: Any) -> Any:
        try:
            return self._ensure().reason(capability, **kwargs)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("world model reason(%s) failed: %s", capability, exc)
            return {"ok": False, "error": str(exc), "capability": capability}


class MemoryAdapter:
    """Bridges the platform to :class:`MemorySystem`."""

    def __init__(self, memory: Any = None) -> None:
        self.memory = memory

    def _ensure(self) -> Any:
        if self.memory is None:
            from cadgenesis.memory import MemorySystem

            self.memory = MemorySystem()
        return self.memory

    def remember(self, pool: str, key: str, content: Any) -> Any:
        return self._ensure().remember(pool, key, content)

    def recall(self, pool: str, key: str) -> Any:
        return self._ensure().recall(pool, key)

    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        result = self._ensure().retrieve(query, top_k=top_k)
        return [{"pool": hit.pool, "score": hit.score, "key": hit.entry.key} for hit in result.hits]


class ReasoningAdapter:
    """Bridges the platform to the reasoning engines."""

    def __init__(self, planner: Any = None, solver: Any = None) -> None:
        from cadgenesis.reasoning import ConstraintSolver, TaskPlanner

        self.planner = planner or TaskPlanner()
        self.solver = solver or ConstraintSolver()

    def create_plan(self, goal: str) -> Any:
        return self.planner.create_plan(goal)

    def solve(self, variables: list[Any], constraints: list[Any]) -> Any:
        return self.solver.solve(variables, constraints)


class ExecutionAdapter:
    """Bridges the platform to :class:`CADExecutionEngine`."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine

    def _ensure(self) -> Any:
        if self.engine is None:
            from cadgenesis.execution import CADExecutionEngine

            self.engine = CADExecutionEngine()
        return self.engine

    def execute(self, cad_tokens: list[str]) -> Any:
        return self._ensure().execute_and_evaluate(cad_tokens)

    def execute_design(
        self, design: dict[str, Any], export_fmt: str | None = None, **kwargs: Any
    ) -> Any:
        """Execute a design dict through the full Pillar 8 pipeline."""
        return self._ensure().execute(design=design, export_fmt=export_fmt, **kwargs)


class ConfidenceAdapter:
    """Bridges the platform to :class:`ConfidenceEngine` (torch, lazy)."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine

    def _ensure(self) -> Any:
        if self.engine is None:
            from cadgenesis.confidence import ConfidenceEngine

            self.engine = ConfidenceEngine()
        return self.engine

    def score(self, logits: Any, confidence_head: Any) -> tuple[float, float]:
        return self._ensure().compute_sequence_confidence(logits, confidence_head)


class ContinualLearningHooks:
    """Stub-safe hooks for the (not yet implemented) continual-learning modules.

    Each hook checks whether the target module is implemented before calling
    it, so the platform degrades gracefully.
    """

    def _call(self, module_name: str, func_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            import importlib

            module = importlib.import_module(f"cadgenesis.continual_learning.{module_name}")
            func = getattr(module, func_name, None)
            if func is None:
                return None
            return func(*args, **kwargs)
        except (ImportError, AttributeError, TypeError) as exc:
            logger.debug(
                "continual-learning hook %s.%s unavailable: %s", module_name, func_name, exc
            )
            return None

    def record_experience(self, task: str, payload: dict[str, Any]) -> Any:
        return self._call("replay_buffer", "record", task, payload)

    def consolidate(self, model: Any) -> Any:
        return self._call("ewc", "consolidate", model)


class NeuroSymbolicAdapter:
    """Bridges the platform to :class:`NeuroSymbolicReasoningEngine`."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine

    def _ensure(self) -> Any:
        if self.engine is None:
            from cadgenesis.reasoning import NeuroSymbolicReasoningEngine

            self.engine = NeuroSymbolicReasoningEngine()
        return self.engine

    def reason(self, symbolic_facts: Any, neural_state: Any) -> Any:
        return self._ensure().forward(symbolic_facts, neural_state)


class PlatformIntegrations:
    """All P5 integration adapters bundled behind one facade."""

    def __init__(
        self,
        transformer: TransformerAgentAdapter | None = None,
        tokenizer: TokenizerAdapter | None = None,
        world_model: WorldModelAdapter | None = None,
        memory: MemoryAdapter | None = None,
        reasoning: ReasoningAdapter | None = None,
        execution: ExecutionAdapter | None = None,
        confidence: ConfidenceAdapter | None = None,
        neuro_symbolic: NeuroSymbolicAdapter | None = None,
    ) -> None:
        self.transformer = transformer or TransformerAgentAdapter()
        self.tokenizer = tokenizer or TokenizerAdapter()
        self.world_model = world_model or WorldModelAdapter()
        self.memory = memory or MemoryAdapter()
        self.reasoning = reasoning or ReasoningAdapter()
        self.execution = execution or ExecutionAdapter()
        self.confidence = confidence or ConfidenceAdapter()
        self.neuro_symbolic = neuro_symbolic or NeuroSymbolicAdapter()
        self.continual = ContinualLearningHooks()

    def status(self) -> dict[str, bool]:
        return {
            "tokenizer_ready": self.tokenizer.tokenizer is not None,
            "world_model_ready": self.world_model.system is not None,
            "memory_ready": self.memory.memory is not None,
            "transformer_ready": self.transformer._block is not None,
            "execution_ready": self.execution.engine is not None,
            "reasoning_ready": self.reasoning.planner is not None,
        }
