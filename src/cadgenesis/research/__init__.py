"""
cadgenesis.research
===================
Research infrastructure for CADGenesis-LM (Pillar 12).

- experiments: experiment tracking + hyperparameter tracking
- datasets: dataset versioning, lineage, integrity verification
- stats: confidence intervals, hypothesis tests, effect sizes
- benchmarks: 6 builtin suites (CAD generation, assembly, reasoning,
  planning, multimodal, manufacturing) + custom suites
- ablation: component/layer/attention/memory/agent ablations
- comparison: variant comparison with significance matrix
- reports: Markdown/HTML/PDF/interactive dashboard report generation
- dashboard: experiment dashboard with embedded charts
- profiler: system + inference + training performance profiling
- reproducibility: seeds, deterministic training, environment capture
- artifacts: artifact registry with content addressing
- integration: ResearchSession + launch_experiment workflow
"""

from cadgenesis.research.artifacts import ArtifactRecord, ArtifactRegistry, file_sha256
from cadgenesis.research.benchmarks import BenchmarkResult, BenchmarkRunner
from cadgenesis.research.comparison import ComparisonReport, ModelComparator
from cadgenesis.research.datasets import DatasetRegistry, DatasetVersion, bump_version, sha256_file
from cadgenesis.research.experiments import (
    ExperimentRecord,
    ExperimentTracker,
    Hyperparams,
    new_experiment_id,
)
from cadgenesis.research.integration import ResearchSession, launch_experiment
from cadgenesis.research.profiler import PerformanceProfiler
from cadgenesis.research.reports import ReportBuilder
from cadgenesis.research.reproducibility import (
    DeterministicTraining,
    EnvironmentCapture,
    SeedRegistry,
    capture_pip_freeze,
    set_seed,
)
from cadgenesis.research.stats import (
    ConfidenceInterval,
    HypothesisTestResult,
    bootstrap_ci,
    cohens_d,
    describe,
    mean_ci_normal,
    welch_t_test,
)

__all__ = [
    "ArtifactRecord",
    "ArtifactRegistry",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ComparisonReport",
    "ConfidenceInterval",
    "DatasetRegistry",
    "DatasetVersion",
    "DeterministicTraining",
    "EnvironmentCapture",
    "ExperimentRecord",
    "ExperimentTracker",
    "Hyperparams",
    "HypothesisTestResult",
    "ModelComparator",
    "PerformanceProfiler",
    "ReportBuilder",
    "ResearchSession",
    "SeedRegistry",
    "bootstrap_ci",
    "bump_version",
    "capture_pip_freeze",
    "cohens_d",
    "describe",
    "file_sha256",
    "launch_experiment",
    "mean_ci_normal",
    "new_experiment_id",
    "set_seed",
    "sha256_file",
    "welch_t_test",
]
