"""Autonomous Engineering Platform (Pillar 20).

The final pillar integrating all previous pillars into one unified engineering
intelligence platform capable of autonomously understanding engineering intent,
reasoning, planning, generating, validating, optimizing, documenting, and
continuously improving CAD solutions.
"""

from __future__ import annotations

from .benchmark import BenchmarkResult, BenchmarkSuite, SystemBenchmark
from .documentation import AutonomousDocumentation, DocumentSet
from .explainability import ExplainableEngineeringAI, ExplanationReport
from .health import HealthMetric, HealthReport, SystemHealthMonitor
from .orchestrator import UnifiedWorkflowOrchestrator, WorkflowStage, WorkflowState
from .plugins import EnterprisePluginFramework, PluginManifest, PluginRegistry
from .validation import EndToEndValidator, ValidationReport, ValidationResult

__all__ = [
    "AutonomousDocumentation",
    "BenchmarkResult",
    "BenchmarkSuite",
    "DocumentSet",
    "EndToEndValidator",
    "EnterprisePluginFramework",
    "ExplainableEngineeringAI",
    "ExplanationReport",
    "HealthMetric",
    "HealthReport",
    "PluginManifest",
    "PluginRegistry",
    "SystemBenchmark",
    "SystemHealthMonitor",
    "UnifiedWorkflowOrchestrator",
    "ValidationReport",
    "ValidationResult",
    "WorkflowStage",
    "WorkflowState",
]
