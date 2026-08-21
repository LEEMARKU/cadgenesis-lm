"""
Explainable Engineering AI - Reasoning trace, decision graph, confidence report, design
rationale, optimization summary, manufacturing report.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""

    step_id: str
    stage: str
    input_summary: str
    reasoning: str
    output_summary: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionNode:
    """A node in the decision graph."""

    node_id: str
    decision: str
    alternatives: list[str]
    rationale: str
    confidence: float
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationReport:
    """Complete explanation report."""

    report_id: str
    workflow_id: str
    reasoning_trace: list[ReasoningStep] = field(default_factory=list)
    decision_graph: list[DecisionNode] = field(default_factory=list)
    confidence_report: dict[str, float] = field(default_factory=dict)
    design_rationale: str = ""
    optimization_summary: str = ""
    manufacturing_report: str = ""
    overall_confidence: float = 0.0
    created_at: float = field(default_factory=time.time)


class ExplainableEngineeringAI:
    """Generates explanations for engineering decisions."""

    def __init__(self):
        self._reports: dict[str, ExplanationReport] = {}
        self._lock = RLock()

    def generate_explanation(
        self,
        workflow_id: str,
        workflow_state: Any,  # WorkflowState
        stage_results: dict[str, Any],
        validation_report: Any,  # ValidationReport
    ) -> ExplanationReport:
        """Generate comprehensive explanation."""
        report = ExplanationReport(
            report_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
        )

        # Build reasoning trace from stage results
        for stage, result in stage_results.items():
            if isinstance(result, dict):
                step = ReasoningStep(
                    step_id=str(uuid.uuid4()),
                    stage=stage,
                    input_summary=result.get("input_summary", "N/A"),
                    reasoning=result.get("reasoning", "N/A"),
                    output_summary=result.get("output_summary", str(result)),
                    confidence=result.get("confidence", 0.5),
                    evidence=result.get("evidence", []),
                )
                report.reasoning_trace.append(step)

        # Build decision graph
        report.decision_graph = self._extract_decisions(stage_results)

        # Confidence report
        report.confidence_report = {
            "generation": stage_results.get("cad_generation", {}).get("confidence", 0.5),
            "validation": validation_report.overall_score if validation_report else 0.5,
            "simulation": stage_results.get("simulation", {}).get("confidence", 0.5),
            "manufacturing": stage_results.get("manufacturing_analysis", {}).get("confidence", 0.5),
        }
        report.overall_confidence = sum(report.confidence_report.values()) / len(
            report.confidence_report
        )

        # Design rationale
        report.design_rationale = self._generate_rationale(stage_results)

        # Optimization summary
        report.optimization_summary = stage_results.get("optimization", {}).get(
            "summary", "No optimization performed"
        )

        # Manufacturing report
        report.manufacturing_report = stage_results.get("manufacturing_analysis", {}).get(
            "report", "No manufacturing analysis"
        )

        with self._lock:
            self._reports[report.report_id] = report

        return report

    def _extract_decisions(self, stage_results: dict[str, Any]) -> list[DecisionNode]:
        decisions = []

        # Extract key decisions from each stage
        decision_stages = [
            "intent_extraction",
            "requirement_graph",
            "planner_agent",
            "cad_generation",
            "optimization",
        ]

        for stage in decision_stages:
            if stage in stage_results:
                result = stage_results[stage]
                if isinstance(result, dict):
                    node = DecisionNode(
                        node_id=str(uuid.uuid4()),
                        decision=result.get("decision", f"Processed {stage}"),
                        alternatives=result.get("alternatives", []),
                        rationale=result.get("rationale", "Automated decision"),
                        confidence=result.get("confidence", 0.5),
                        dependencies=result.get("dependencies", []),
                    )
                    decisions.append(node)

        return decisions

    def _generate_rationale(self, stage_results: dict[str, Any]) -> str:
        rationale_parts = []

        if "requirement_graph" in stage_results:
            req = stage_results["requirement_graph"]
            rationale_parts.append(f"Requirements: {req.get('summary', 'N/A')}")

        if "world_model" in stage_results:
            wm = stage_results["world_model"]
            rationale_parts.append(f"World model reasoning: {wm.get('reasoning', 'N/A')}")

        if "neuro_symbolic_reasoning" in stage_results:
            nsr = stage_results["neuro_symbolic_reasoning"]
            rationale_parts.append(f"Symbolic validation: {nsr.get('validation_summary', 'N/A')}")

        return (
            " | ".join(rationale_parts)
            if rationale_parts
            else "Automated engineering pipeline execution"
        )

    def get_report(self, report_id: str) -> ExplanationReport | None:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self, workflow_id: str | None = None) -> list[ExplanationReport]:
        with self._lock:
            reports = list(self._reports.values())
            if workflow_id:
                reports = [r for r in reports if r.workflow_id == workflow_id]
            return reports
