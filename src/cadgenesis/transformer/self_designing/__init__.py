"""
cadgenesis.transformer.self_designing
===============================
Self-Designing & Dynamic Neural Architecture sub-system for CADGenesis-LM v2.0.

Public API
----------
* ``SelfDesigningTransformer`` — self-designing controller wrapping the
  ``GeometryAwareTransformer`` backbone (kept import-compatible with the
  original ``cadgenesis.transformer.self_designing`` module).
* ``ArchitectureSpec`` / ``ArchitectureSearchSpace`` / ``NeuralArchitectureSearch``
* ``ArchitectureScore`` / ``ArchitectureEvaluator``
* ``DynamicLayerRouter`` / ``AdaptiveAttentionHeadSelector`` / ``LayerPruningController``
* ``AutomaticRollback``
"""

from cadgenesis.transformer.self_designing.adaptive_heads import AdaptiveAttentionHeadSelector
from cadgenesis.transformer.self_designing.architecture import (
    ArchitectureSearchSpace,
    ArchitectureSpec,
    NeuralArchitectureSearch,
)
from cadgenesis.transformer.self_designing.evaluation import (
    ArchitectureEvaluator,
    ArchitectureScore,
)
from cadgenesis.transformer.self_designing.pruning import LayerPruningController
from cadgenesis.transformer.self_designing.rollback import AutomaticRollback
from cadgenesis.transformer.self_designing.routing import DynamicLayerRouter
from cadgenesis.transformer.self_designing.self_designing import SelfDesigningTransformer

__all__ = [
    "AdaptiveAttentionHeadSelector",
    "ArchitectureEvaluator",
    "ArchitectureScore",
    "ArchitectureSearchSpace",
    "ArchitectureSpec",
    "AutomaticRollback",
    "DynamicLayerRouter",
    "LayerPruningController",
    "NeuralArchitectureSearch",
    "SelfDesigningTransformer",
]
