"""cadgenesis.cad.integration.pipeline
====================================
End-to-end CAD intelligence pipeline.

Given a design (dict or CAD package object), the pipeline:

1. validates it (via the validation pipeline),
2. maps it to reasoning inputs (primitives, constraints, DFM part),
3. tokenizes it into a ``CADTokenSequence`` / ``MultiModalBatch``,
4. stores it in CAD memory,
5. returns a full report with validation, tokens and memory entry.

This is the primary entry point linking the new ``cad`` package with the
existing tokenizer / transformer / reasoning / memory subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.integration.memory_bridge import CADMemoryBridge
from cadgenesis.cad.integration.reasoning_bridge import ReasoningBridge
from cadgenesis.cad.integration.tokenizer_bridge import TokenizerBridge
from cadgenesis.cad.integration.transformer_bridge import TransformerBridge
from cadgenesis.cad.validation.pipeline import CadValidator
from cadgenesis.cad.validation.report import CadValidationReport


@dataclass
class PipelineResult:
    """Outcome of running the CAD intelligence pipeline on one design."""

    name: str
    validation: CadValidationReport | None = None
    tokens: list[str] = field(default_factory=list)
    sequence: Any = None
    batch: Any = None
    memory_key: str | None = None
    reasoning: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        validation = self.validation.summary() if self.validation else None
        return {
            "name": self.name,
            "validation": validation,
            "token_count": len(self.tokens),
            "memory_key": self.memory_key,
            "reasoning": self.reasoning,
            "execution": self.execution,
        }


class CADIntelligencePipeline:
    """Orchestrates validate -> reason -> tokenize -> memorise for a design.

    Optionally runs the Pillar 8 execution pipeline (``execution=True``):
    validate -> simulate -> optimize -> repair -> export -> feedback over the
    analytic CAD substrate, recording the engine summary on the result.
    """

    def __init__(
        self,
        tokenizer=None,
        validator: CadValidator | None = None,
        memory: CADMemoryBridge | None = None,
        remember: bool = True,
        execution: bool = False,
    ) -> None:
        self.validator = validator or CadValidator()
        self.memory = memory or CADMemoryBridge()
        self.remember = remember
        self.execution = execution
        self.tokens: TokenizerBridge | None
        self.transformer: TransformerBridge | None
        if tokenizer is not None:
            self.tokens = TokenizerBridge(tokenizer)
            self.transformer = TransformerBridge(tokenizer)
        else:
            self.tokens = None
            self.transformer = None

    def run(self, design: dict[str, Any], name: str = "design", text: str = "") -> PipelineResult:
        """Process a single design dict through the full pipeline."""
        result = PipelineResult(name=name)

        # 1. validate
        result.validation = self.validator.validate(design)

        # 1b. execute (flag-gated, Pillar 8)
        if self.execution:
            from cadgenesis.execution.execution_engine import CADExecutionEngine

            summary = CADExecutionEngine().execute(design=design).summary()
            result.execution = {
                k: summary[k] for k in ("is_valid_geometry", "is_manufacturable", "safety_factor")
            }

        # 2. reasoning inputs
        primitives = design.get("primitives", [])
        if primitives:
            converted = [ReasoningBridge.to_primitive(p) for p in primitives]
            result.reasoning["primitives"] = [
                {
                    "kind": p.kind,
                    "dims": dict(p.dims),
                    "position": p.position or (0.0, 0.0, 0.0),
                }
                for p in converted
            ]
        part = design.get("part")
        if part is not None:
            result.reasoning["part"] = part

        # 3. tokenize
        if self.tokens is not None:
            result.tokens = self.tokens.design_to_tokens(design)
            result.sequence = self.tokens.to_sequence(design, text=text)
            if self.transformer is not None and text:
                result.batch = self.transformer.encode_batch([design], [text])

        # 4. memorise
        if self.remember and design:
            entry = self.memory.store_design(name, design)
            result.memory_key = entry.key

        return result

    def run_batch(
        self, designs: list[dict[str, Any]], texts: list[str] | None = None
    ) -> list[PipelineResult]:
        """Process many designs; returns one ``PipelineResult`` each."""
        if texts is None:
            texts = [""] * len(designs)
        return [
            self.run(design, name=f"design-{i}", text=text)
            for i, (design, text) in enumerate(zip(designs, texts, strict=False))
        ]


__all__ = ["CADIntelligencePipeline", "PipelineResult"]
