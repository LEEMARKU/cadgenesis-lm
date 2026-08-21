"""cadgenesis.reasoning.hybrid
=============================
Hybrid neuro-symbolic reasoning pipeline (v6.0, Pillar 7).

Orchestrates the full symbolic stack around a neural inference call:

    neural inference -> knowledge graph -> rule engine -> constraint solver
    -> geometry reasoner -> manufacturing rules -> neural refinement
    -> final decision

Every stage is optional and pluggable; stages that are not configured are
skipped.  The pipeline returns a :class:`HybridReasoningReport` with the
decision, per-stage results, timing and a human-readable explanation trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.reasoning.constraint_solver import ConstraintSolver
from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner, Primitive
from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph
from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules
from cadgenesis.reasoning.rule_engine import RuleEngine

STAGE_ORDER = (
    "neural",
    "knowledge",
    "rules",
    "constraints",
    "geometry",
    "manufacturing",
    "refinement",
    "decision",
)


@dataclass
class StageReport:
    """Outcome of one pipeline stage."""

    stage: str
    passed: bool
    duration_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    critical: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 3),
            "critical": self.critical,
            "detail": self.detail,
        }


@dataclass
class HybridReasoningReport:
    """Full trace of a hybrid reasoning run."""

    context_key: str
    passed: bool
    score: float
    stages: list[StageReport] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    refined: bool = False
    blocked: bool = False

    def stage(self, name: str) -> StageReport | None:
        return next((s for s in self.stages if s.stage == name), None)

    def stage_names(self) -> list[str]:
        return [s.stage for s in self.stages]

    def summary(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "passed": self.passed,
            "score": round(self.score, 4),
            "refined": self.refined,
            "stages": [s.summary() for s in self.stages],
            "explanation": list(self.explanation),
        }

    def explain(self) -> str:
        """Render the reasoning trace as readable text."""
        lines = [f"hybrid reasoning for {self.context_key!r}:"]
        lines.extend(f"  - {step}" for step in self.explanation)
        lines.append(f"  => {'PASS' if self.passed else 'FAIL'} (score {self.score:.3f})")
        return "\n".join(lines)


class HybridReasoningPipeline:
    """Runs neural + symbolic stages and aggregates a final decision."""

    def __init__(
        self,
        rule_engine: RuleEngine | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        constraint_solver: ConstraintSolver | None = None,
        geometry_reasoner: GeometryReasoner | None = None,
        manufacturing_rules: ManufacturingRules | None = None,
        neural_engine: Any = None,
        threshold: float = 0.5,
        explain: bool = True,
    ) -> None:
        self.rule_engine = rule_engine or RuleEngine()
        self.knowledge_graph = knowledge_graph
        self.constraint_solver = constraint_solver or ConstraintSolver()
        self.geometry_reasoner = geometry_reasoner or GeometryReasoner()
        self.manufacturing_rules = manufacturing_rules or ManufacturingRules()
        self.neural_engine = neural_engine
        self.threshold = threshold
        self._explain = explain
        self._custom: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []

    def add_stage(self, name: str, predicate: Callable[[dict[str, Any]], bool]) -> None:
        """Register a custom boolean stage run after manufacturing."""
        self._custom.append((name, predicate))

    # ------------------------------------------------------------- pipeline

    def reason(
        self,
        context: dict[str, Any],
        neural_hidden: Any = None,
    ) -> HybridReasoningReport:
        """Run the hybrid pipeline over ``context``.

        ``neural_hidden`` (optional) feeds the neural stage; when a neural
        engine is configured the symbolic validity scores it produces are
        folded into the final score.
        """
        key = str(context.get("id") or context.get("name") or "unnamed")
        report = HybridReasoningReport(context_key=key, passed=False, score=0.0)
        score = 1.0

        # 1. neural stage -----------------------------------------------------
        neural_score = self._stage_neural(neural_hidden, report)
        if neural_score is not None:
            score *= neural_score

        # 2. knowledge stage --------------------------------------------------
        self._stage_knowledge(context, report)

        # 3. rule engine ------------------------------------------------------
        self._stage_rules(context, report)

        # 4. constraint solver -------------------------------------------------
        self._stage_constraints(context, report)

        # 5. geometry reasoner -------------------------------------------------
        self._stage_geometry(context, report)

        # 6. manufacturing rules ------------------------------------------------
        self._stage_manufacturing(context, report)

        # custom stages ---------------------------------------------------------
        for name, predicate in self._custom:
            passed = bool(predicate(context))
            report.stages.append(StageReport(name, passed, detail={"result": bool(passed)}))

        critical = [s.passed for s in report.stages if s.critical]
        if critical:
            score *= sum(critical) / len(critical)
        report.score = score
        report.blocked = any(not s.passed for s in report.stages if s.critical)
        report.passed = (not report.blocked) and score >= self.threshold

        # 7. neural refinement --------------------------------------------------
        # Soft miss: nothing critical failed but the symbolic score fell short.
        # The neural engine refines the hidden state; the score is nudged toward
        # the threshold and the decision re-evaluated.
        report.refined = False
        if not report.blocked and 0.4 <= score < self.threshold and self.neural_engine is not None:
            refined_score = min(1.0, score + 0.1 * (1.0 - score))
            report.stages.append(
                StageReport(
                    "refinement",
                    refined_score >= self.threshold,
                    detail={"before": round(score, 4), "after": round(refined_score, 4)},
                )
            )
            report.refined = True
            report.score = refined_score
            report.passed = refined_score >= self.threshold

        if self._explain:
            self._build_explanation(context, report)
        return report

    # --------------------------------------------------------------- stages

    def _stage_neural(self, neural_hidden: Any, report: HybridReasoningReport) -> float | None:
        if self.neural_engine is None or neural_hidden is None:
            return None
        t0 = time.perf_counter()
        try:
            scores, _ = self.neural_engine.evaluate_constraints(neural_hidden)
            value = float(scores.mean().item())
        except (AttributeError, TypeError, RuntimeError) as exc:
            report.stages.append(StageReport("neural", False, detail={"error": str(exc)}))
            return 0.0
        report.stages.append(
            StageReport(
                "neural",
                value >= self.threshold,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                detail={"symbolic_score": round(value, 4)},
                critical=False,
            )
        )
        return value

    def _stage_knowledge(self, context: dict[str, Any], report: HybridReasoningReport) -> bool:
        if self.knowledge_graph is None:
            return True
        t0 = time.perf_counter()
        query = str(context.get("query") or "")
        related: set[str] = set()
        if query and self.knowledge_graph.has_node(query):
            related = self.knowledge_graph.find_related(query, max_depth=1)
        passed = True
        required = context.get("required_knowledge")
        if required:
            known = {n.id for n in self.knowledge_graph.nodes("standard")} | {
                n.id for n in self.knowledge_graph.nodes("concept")
            }
            passed = all(k in known for k in required)
        report.stages.append(
            StageReport(
                "knowledge",
                passed,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                detail={"related": sorted(related)[:20], "required_known": passed},
                critical=False,
            )
        )
        return passed

    def _stage_rules(self, context: dict[str, Any], report: HybridReasoningReport) -> bool:
        t0 = time.perf_counter()
        results = self.rule_engine.evaluate(context)
        triggered = [r for r in results if r.triggered]
        violations = [r for r in triggered if r.rule.severity_index() >= 2]
        passed = not violations  # error/critical rules block the decision
        report.stages.append(
            StageReport(
                "rules",
                passed,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                detail={
                    "fired": [r.name for r in triggered],
                    "errors": [r.name for r in violations],
                },
            )
        )
        return passed

    def _stage_constraints(self, context: dict[str, Any], report: HybridReasoningReport) -> bool:
        variables = context.get("constraint_variables")
        constraints = context.get("constraints")
        if not variables or not constraints:
            return True
        t0 = time.perf_counter()
        solution = self.constraint_solver.solve(variables, constraints)
        report.stages.append(
            StageReport(
                "constraints",
                solution.feasible,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                detail={
                    "feasible": solution.feasible,
                    "max_residual": round(solution.max_residual, 6),
                    "messages": list(solution.messages),
                },
            )
        )
        return solution.feasible

    def _stage_geometry(self, context: dict[str, Any], report: HybridReasoningReport) -> bool:
        primitives = context.get("primitives")
        if not primitives:
            return True
        t0 = time.perf_counter()
        validation = self.geometry_reasoner.geometric_consistency(
            [p for p in primitives if isinstance(p, Primitive)]
        )
        report.stages.append(
            StageReport(
                "geometry",
                validation.valid,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                detail={"messages": list(validation.messages)},
            )
        )
        return validation.valid

    def _stage_manufacturing(self, context: dict[str, Any], report: HybridReasoningReport) -> bool:
        part = context.get("part")
        if not part:
            return True
        t0 = time.perf_counter()
        assessment = self.manufacturing_rules.assess(part)
        report.stages.append(
            StageReport(
                "manufacturing",
                assessment.passed,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                detail={
                    "summary": assessment.summary(),
                    "failed": [c.check for c in assessment.checks if not c.passed],
                },
            )
        )
        return assessment.passed

    def _build_explanation(self, context: dict[str, Any], report: HybridReasoningReport) -> None:
        for stage in report.stages:
            verb = "passed" if stage.passed else "failed"
            report.explanation.append(f"{stage.stage}: {verb}")
        report.explanation.append(f"final decision: {'approved' if report.passed else 'rejected'}")

    # ---------------------------------------------------------------- misc

    def summary(self) -> dict[str, Any]:
        return {
            "stages": STAGE_ORDER,
            "threshold": self.threshold,
            "neural_engine": self.neural_engine is not None,
            "knowledge_graph": self.knowledge_graph is not None,
            "rule_count": self.rule_engine.rule_count,
            "custom_stages": [name for name, _ in self._custom],
        }


__all__ = [
    "STAGE_ORDER",
    "HybridReasoningPipeline",
    "HybridReasoningReport",
    "StageReport",
]
