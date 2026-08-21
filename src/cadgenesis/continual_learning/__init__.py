"""cadgenesis.continual_learning
=============================
Continual Learning subsystem: replay buffers, EWC regularization, task-isolated
adapters, knowledge anchors, continual training, evaluation and checkpoint
updating.
"""

from __future__ import annotations

from cadgenesis.continual_learning.adapter_isolation import TaskAdapterRegistry, TaskIsolation
from cadgenesis.continual_learning.continual_trainer import ContinualTrainer
from cadgenesis.continual_learning.evaluator import ContinualEvaluator
from cadgenesis.continual_learning.ewc import EWC
from cadgenesis.continual_learning.knowledge_anchor import KnowledgeAnchor
from cadgenesis.continual_learning.replay_buffer import ReplayBuffer, ReplaySample
from cadgenesis.continual_learning.updater import ModelUpdater

__all__ = [
    "EWC",
    "ContinualEvaluator",
    "ContinualTrainer",
    "KnowledgeAnchor",
    "ModelUpdater",
    "ReplayBuffer",
    "ReplaySample",
    "TaskAdapterRegistry",
    "TaskIsolation",
]
