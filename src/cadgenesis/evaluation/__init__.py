"""cadgenesis.evaluation
=====================
Metrics, evaluation harnesses, and report generation for CADGenesis models.
"""

from cadgenesis.evaluation.agent_metrics import (
    consensus_agreement,
    error_rate,
    fleet_coverage,
    mean_latency,
    p95_latency,
    pipeline_success,
    run_agent_benchmark,
    task_success_rate,
)
from cadgenesis.evaluation.benchmark_runner import BenchmarkSummary, run_pillar_benchmark
from cadgenesis.evaluation.cad_bench import (
    CADBenchItem,
    CADBenchmark,
    CADBenchResult,
)
from cadgenesis.evaluation.cad_metrics import CADMetrics
from cadgenesis.evaluation.execution_metrics import (
    confidence_agreement,
    cost_error,
    geometry_validity_rate,
    manufacturability_rate,
    repair_rate,
    run_execution_benchmark,
    safety_factor_pass_rate,
    simulation_pass_rate,
)
from cadgenesis.evaluation.geometry_metrics import GeometryMetrics
from cadgenesis.evaluation.memory_metrics import (
    compression_ratio,
    consolidation_ratio,
    memory_hit_rate,
    precision_at_k,
    routing_accuracy,
    run_memory_benchmark,
)
from cadgenesis.evaluation.memory_metrics import (
    mean_reciprocal_rank as memory_mean_reciprocal_rank,
)
from cadgenesis.evaluation.memory_metrics import (
    recall_at_k as memory_recall_at_k,
)
from cadgenesis.evaluation.multimodal_metrics import (
    MultimodalMetrics,
    alignment_metrics,
    cross_modal_retrieval,
    evaluate_retrieval,
    fusion_intra_inter_ratio,
    mean_reciprocal_rank,
    recall_at_k,
    run_retrieval_benchmark,
)
from cadgenesis.evaluation.reasoning_metrics import (
    constraint_reasoning,
    engineering_correctness,
    manufacturing_correctness,
    reasoning_accuracy,
    rule_utilization,
    run_reasoning_benchmark,
    symbolic_consistency,
    topology_reasoning,
)
from cadgenesis.evaluation.report_generator import ReportGenerator, generate_report
from cadgenesis.evaluation.tokenizer_metrics import TokenizerMetrics
from cadgenesis.evaluation.world_model_metrics import (
    accuracy,
    affordance_coverage_with,
    assembly_integrity,
    path_collision_detection,
    planning_success,
    run_world_benchmark,
    safety_margin,
)

__all__ = [
    "BenchmarkSummary",
    "CADBenchItem",
    "CADBenchResult",
    "CADBenchmark",
    "CADMetrics",
    "GeometryMetrics",
    "MultimodalMetrics",
    "ReportGenerator",
    "TokenizerMetrics",
    "accuracy",
    "affordance_coverage_with",
    "alignment_metrics",
    "assembly_integrity",
    "compression_ratio",
    "confidence_agreement",
    "consensus_agreement",
    "consolidation_ratio",
    "constraint_reasoning",
    "cost_error",
    "cross_modal_retrieval",
    "engineering_correctness",
    "error_rate",
    "evaluate_retrieval",
    "fleet_coverage",
    "fusion_intra_inter_ratio",
    "generate_report",
    "geometry_validity_rate",
    "manufacturability_rate",
    "manufacturing_correctness",
    "mean_latency",
    "mean_reciprocal_rank",
    "memory_hit_rate",
    "memory_mean_reciprocal_rank",
    "memory_recall_at_k",
    "p95_latency",
    "path_collision_detection",
    "pipeline_success",
    "planning_success",
    "precision_at_k",
    "reasoning_accuracy",
    "recall_at_k",
    "repair_rate",
    "routing_accuracy",
    "rule_utilization",
    "run_agent_benchmark",
    "run_execution_benchmark",
    "run_memory_benchmark",
    "run_pillar_benchmark",
    "run_reasoning_benchmark",
    "run_retrieval_benchmark",
    "run_world_benchmark",
    "safety_factor_pass_rate",
    "safety_margin",
    "simulation_pass_rate",
    "symbolic_consistency",
    "task_success_rate",
    "topology_reasoning",
]
