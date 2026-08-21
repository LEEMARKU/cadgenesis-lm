"""cadgenesis.cad.integration.execution_bridge
============================================
Bridge from the ``cad`` package to the execution engine
(:mod:`cadgenesis.execution.execution_engine`).

The execution engine consumes a CAD token stream and returns
``CADExecutionResult`` (geometry validity, manufacturability, cost, safety
factor, confidence and feedback suggestions).  This bridge converts CAD objects
(design dicts, feature trees, meshes) into the token stream the engine
expects, using :class:`TokenizerBridge`, then runs the engine and re-exposes
the result with a small domain wrapper.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.integration.tokenizer_bridge import TokenizerBridge
from cadgenesis.execution.execution_engine import CADExecutionEngine, CADExecutionResult


class ExecutionBridge:
    """Run the CAD execution engine over CAD objects or token streams."""

    def __init__(self, engine: CADExecutionEngine | None = None) -> None:
        self.engine = engine or CADExecutionEngine()

    def run_tokens(self, cad_tokens: list[str]) -> CADExecutionResult:
        """Execute a raw CAD token stream and return the engine result."""
        return self.engine.execute_and_evaluate(list(cad_tokens))

    def run_design(
        self,
        design: dict[str, Any],
        tokenizer_bridge: TokenizerBridge,
    ) -> CADExecutionResult:
        """Tokenize a design dict and execute it."""
        cad_tokens = tokenizer_bridge.design_to_tokens(design)
        return self.engine.execute_and_evaluate(cad_tokens)

    def execute_design(
        self,
        design: dict[str, Any],
        export_fmt: str | None = None,
        export_path: str | None = None,
        memory: Any = None,
    ) -> CADExecutionResult:
        """Execute a design dict through the full pipeline (additive path).

        Runs the real engine pipeline: validate -> simulate -> optimize ->
        repair -> export -> feedback, plus optional memory persistence.
        """
        return self.engine.execute(
            design=design,
            export_fmt=export_fmt,
            export_path=export_path,
            memory=memory,
        )

    def summary(self, result: CADExecutionResult) -> dict[str, Any]:
        """Compact, JSON-friendly summary of an execution result."""
        return {
            "is_valid_geometry": result.is_valid_geometry,
            "is_manufacturable": result.is_manufacturable,
            "safety_factor": result.safety_factor,
            "estimated_cost_usd": result.estimated_cost_usd,
            "confidence_score": result.confidence_score,
            "errors": list(result.errors),
            "suggestions": list(result.suggestions),
        }


__all__ = ["ExecutionBridge"]
