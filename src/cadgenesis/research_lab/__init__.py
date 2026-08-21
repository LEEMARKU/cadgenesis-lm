"""Frontier AI Research Laboratory (Pillar 16).

Isolated research environment for developing, evaluating, and benchmarking
new AI ideas without affecting the production model. Experimental modules
remain sandboxed until validated.
"""

from __future__ import annotations

from .agent_lab import AgentResearchLab
from .evaluation import (
    ABTestConfig,
    ABTestResult,
    EvaluationFramework,
    StatisticalTestConfig,
    StatisticalTestResult,
)
from .experimental_transformer import ExperimentalTransformerLab
from .learning_lab import LearningResearchLab
from .memory_lab import MemoryResearchLab
from .multimodal_lab import MultimodalResearchLab
from .neuro_symbolic_lab import NeuroSymbolicResearchLab
from .promotion import PromotionDecision, PromotionStage, SafePromotionPipeline
from .registry import ExperimentConfig, ExperimentRegistry, ExperimentResult
from .world_model_lab import WorldModelResearchLab

__all__ = [
    "ABTestConfig",
    "ABTestResult",
    "AgentResearchLab",
    "EvaluationFramework",
    "ExperimentConfig",
    "ExperimentRegistry",
    "ExperimentResult",
    "ExperimentalTransformerLab",
    "LearningResearchLab",
    "MemoryResearchLab",
    "MultimodalResearchLab",
    "NeuroSymbolicResearchLab",
    "PromotionDecision",
    "PromotionStage",
    "SafePromotionPipeline",
    "StatisticalTestConfig",
    "StatisticalTestResult",
    "WorldModelResearchLab",
]
