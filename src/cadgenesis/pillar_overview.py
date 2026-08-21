"""High-level pillar overview helpers for CADGenesis-LM.

This module exposes a compact, programmatic summary of the architecture that is
already implemented across the repository. It is intentionally lightweight and
import-safe so it can be used by CLI tools, examples, and tests without
requiring heavyweight runtime dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PillarSummary:
    """A concise summary of a pillar and its current implementation status."""

    name: str
    focus: str
    status: str = "implemented"
    highlights: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_pillar_overview() -> list[dict[str, Any]]:
    """Return a structured overview for all major architecture pillars."""

    pillars = [
        PillarSummary(
            name="Pillar 1",
            focus="Foundation model",
            highlights=(
                "Geometry-aware transformer",
                "Sparse attention",
                "Multi-scale attention",
                "Hierarchical transformer",
                "Mixture-of-experts",
                "Dynamic routing",
            ),
        ),
        PillarSummary(
            name="Pillar 2",
            focus="CAD intelligence",
            highlights=(
                "Parametric modeling",
                "Feature-based modeling",
                "B-Rep and CSG support",
                "Assembly and mechanism modeling",
            ),
        ),
        PillarSummary(
            name="Pillar 3",
            focus="Multimodal understanding",
            highlights=(
                "Shared embedding space",
                "Cross-modal attention",
                "Multimodal fusion",
                "CAD/text/image/sketch support",
            ),
        ),
        PillarSummary(
            name="Pillar 4",
            focus="World model",
            highlights=(
                "Spatial and mechanical reasoning",
                "Assembly and functional reasoning",
                "Design intent modeling",
                "Simulation integration",
            ),
        ),
        PillarSummary(
            name="Pillar 5",
            focus="Multi-agent intelligence",
            highlights=(
                "Agent infrastructure",
                "Scheduling and consensus",
                "Shared memory",
                "Specialized engineering agents",
            ),
        ),
        PillarSummary(
            name="Pillar 6",
            focus="Layer-integrated memory",
            highlights=(
                "Working/session/project memory",
                "Retrieval and routing",
                "Compression and persistence",
                "Memory-augmented inference",
            ),
        ),
        PillarSummary(
            name="Pillar 7",
            focus="Neuro-symbolic reasoning",
            highlights=(
                "Knowledge graph",
                "Rule engine",
                "Constraint and topology reasoning",
                "Manufacturing standards",
            ),
        ),
        PillarSummary(
            name="Pillar 8",
            focus="CAD execution and validation",
            highlights=(
                "Execution engine",
                "Geometry and topology validation",
                "Manufacturing checks",
                "Optimization and export",
            ),
        ),
        PillarSummary(
            name="Pillar 9",
            focus="Learning system",
            highlights=(
                "Training infrastructure",
                "Continual learning",
                "Distillation and PEFT",
                "Self-improvement hooks",
            ),
        ),
        PillarSummary(
            name="Pillar 10",
            focus="Reliability and confidence AI",
            highlights=(
                "Confidence estimation",
                "Uncertainty estimation",
                "Hallucination checks",
                "Auto-repair and fallback",
            ),
        ),
        PillarSummary(
            name="Pillar 11",
            focus="Production platform",
            highlights=(
                "REST/gRPC/SDK/CLI",
                "Authentication and monitoring",
                "Plugin and config systems",
            ),
        ),
        PillarSummary(
            name="Pillar 12",
            focus="Research infrastructure",
            highlights=(
                "Experiment tracking",
                "Benchmarking and ablations",
                "Reproducibility tooling",
                "Artifact registry",
            ),
        ),
        PillarSummary(
            name="Pillar 13",
            focus="Engineering trust infrastructure",
            highlights=(
                "Provenance tracking",
                "Integrity and signatures",
                "Versioning and rollback",
                "Trust layer abstraction",
            ),
        ),
        PillarSummary(
            name="Pillar 14",
            focus="Collaborative research economy",
            highlights=(
                "Plugin and adapter sharing",
                "Benchmark and contribution tracking",
                "Collaboration registry",
            ),
        ),
        PillarSummary(
            name="Pillar 15",
            focus="Advanced optimization & HPC engine",
            highlights=(
                "Distributed training/inference",
                "Compiler optimizations",
                "Memory optimization",
                "Performance dashboard",
            ),
        ),
        PillarSummary(
            name="Pillar 16",
            focus="Frontier AI research laboratory",
            highlights=(
                "Experimental transformer lab",
                "Memory/multimodal/world model labs",
                "Agent/neuro-symbolic/learning labs",
                "Evaluation framework & promotion pipeline",
            ),
        ),
        PillarSummary(
            name="Pillar 17",
            focus="Autonomous AI research laboratory",
            highlights=(
                "Research planner & hypothesis generator",
                "Automated experiment runner",
                "Statistical analyzer & hyperparameter search",
                "Human approval pipeline",
            ),
        ),
        PillarSummary(
            name="Pillar 18",
            focus="Global engineering knowledge network",
            highlights=(
                "Knowledge graph & standards library",
                "Material & manufacturing databases",
                "Hybrid retrieval & RAG",
                "Enterprise knowledge connectors",
            ),
        ),
        PillarSummary(
            name="Pillar 19",
            focus="Industrial digital twin",
            highlights=(
                "Product/factory/machine/process twins",
                "Real-time synchronization",
                "Predictive analytics",
                "Lifecycle management",
            ),
        ),
        PillarSummary(
            name="Pillar 20",
            focus="Autonomous engineering platform",
            highlights=(
                "Unified 22-stage workflow orchestrator",
                "End-to-end validation",
                "Explainable engineering AI",
                "Autonomous documentation & health monitoring",
            ),
        ),
    ]

    return [pillar.to_dict() for pillar in pillars]
