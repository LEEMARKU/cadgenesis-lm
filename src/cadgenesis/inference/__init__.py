"""
cadgenesis.inference
====================
Inference subsystem for CADGenesis-LM v2.0.

Production decoding of a natural-language design request into a CAD token
sequence.  ``cadgenesis.inference.engine`` hosts the
:class:`~cadgenesis.inference.engine.CADInferenceEngine`; ``serving`` builds
on top of this for HTTP / model-serving deployment.

Test-time compute and speculative decoding are exposed here too:
* :func:`best_of_n`, :func:`self_consistency`, :func:`mcts` — oracle-driven
  search that spends more compute at inference time.
* :class:`EagleDraftHead`, :func:`train_eagle`, :func:`speculative_eagle` —
  EAGLE-style learned speculative decoding (greedy-preserving).
"""

from cadgenesis.inference.eagle import (
    EagleDraftHead,
    collect_hidden_pairs,
    speculative_eagle,
    train_eagle,
)
from cadgenesis.inference.engine import CADInferenceEngine, GenerationResult
from cadgenesis.inference.mcts import best_of_n, mcts, self_consistency

__all__ = [
    "CADInferenceEngine",
    "EagleDraftHead",
    "GenerationResult",
    "best_of_n",
    "collect_hidden_pairs",
    "mcts",
    "self_consistency",
    "speculative_eagle",
    "train_eagle",
]
