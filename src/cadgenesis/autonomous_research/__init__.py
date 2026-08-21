"""Autonomous AI Research Laboratory (Pillar 17).

AI Research Laboratory capable of automatically designing, executing, evaluating,
and documenting machine learning experiments while keeping humans in control of approvals.
"""

from __future__ import annotations

from .approval import ApprovalDecision, ApprovalRequest, HumanApprovalPipeline
from .comparator import ArchitectureComparator, ComparisonResult
from .evaluator import BenchmarkEvaluator, EvaluationReport
from .experiment_planner import ExperimentGraph, ExperimentNode, ExperimentPlanner
from .failure_analyzer import FailureAnalyzer, FailureReport
from .hyperparameter_search import HyperparameterSearch, SearchResult, SearchSpace
from .hypothesis import Hypothesis, HypothesisGenerator
from .planner import ResearchObjective, ResearchPlan, ResearchPlanner
from .report_generator import ReportFormat, ResearchReportGenerator
from .runner import AutomatedExperimentRunner, ExperimentExecution
from .statistics import StatisticalAnalyzer, StatisticalReport

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ArchitectureComparator",
    "AutomatedExperimentRunner",
    "BenchmarkEvaluator",
    "ComparisonResult",
    "EvaluationReport",
    "ExperimentExecution",
    "ExperimentGraph",
    "ExperimentNode",
    "ExperimentPlanner",
    "FailureAnalyzer",
    "FailureReport",
    "HumanApprovalPipeline",
    "HyperparameterSearch",
    "Hypothesis",
    "HypothesisGenerator",
    "ReportFormat",
    "ResearchObjective",
    "ResearchPlan",
    "ResearchPlanner",
    "ResearchReportGenerator",
    "SearchResult",
    "SearchSpace",
    "StatisticalAnalyzer",
    "StatisticalReport",
]
