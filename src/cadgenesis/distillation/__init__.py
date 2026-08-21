"""
cadgenesis.distillation
=======================
Multi-Teacher Distillation Engine for CADGenesis-LM v2.0.

Also exports the P2 post-training trainers:
* :class:`GRPOTrainer` — GRPO (group-relative policy optimization) fine-tuning
  with a validity-based reward (DeepSeek-R1 style).
* :class:`DPOTrainer` — direct preference optimization (Rafailov et al., 2023).
* :class:`RLVRTrainer` — RLVR (verifiable-reward) GRPO trainer.

And the distillation support modules (v6.0):
* :class:`SoftLabelGenerator` — temperature-scaled soft targets and KL loss.
* :class:`HardLabelExtractor` — filtered hard labels from teacher logits.
* :class:`TeacherConsensus` — weighted TOON-vote consensus + logit consensus.
* :class:`CritiqueEngine` — rule-based TOON critiquer for self-improvement.
* :class:`RLAIFEngine` — Bradley-Terry preference signals from critiques.
* :class:`SyntheticDataGenerator` — deterministic rule-based (prompt, TOON)
  sample generation.
* :class:`DistillationPipeline` — end-to-end distillation orchestration.
"""

from cadgenesis.distillation.consensus import ConsensusResult, TeacherConsensus
from cadgenesis.distillation.critique import CritiqueEngine, CritiqueFeedback
from cadgenesis.distillation.distillation_engine import (
    MultiTeacherDistillationEngine,
    TeacherConsensusEngine,
)
from cadgenesis.distillation.dpo import DPOTrainer
from cadgenesis.distillation.grpo import GRPOTrainer
from cadgenesis.distillation.hard_labels import HardLabelBatch, HardLabelExtractor
from cadgenesis.distillation.pipeline import DistillationPipeline, DistillationRunReport
from cadgenesis.distillation.rlaif import RLAIFEngine
from cadgenesis.distillation.rlvr import (
    DesignOracle,
    MockOracle,
    RLVRTrainer,
    VerifiableOracle,
)
from cadgenesis.distillation.soft_labels import SoftLabelGenerator
from cadgenesis.distillation.synthetic import SyntheticDataGenerator

__all__ = [
    "ConsensusResult",
    "CritiqueEngine",
    "CritiqueFeedback",
    "DPOTrainer",
    "DesignOracle",
    "DistillationPipeline",
    "DistillationRunReport",
    "GRPOTrainer",
    "HardLabelBatch",
    "HardLabelExtractor",
    "MockOracle",
    "MultiTeacherDistillationEngine",
    "RLAIFEngine",
    "RLVRTrainer",
    "SoftLabelGenerator",
    "SyntheticDataGenerator",
    "TeacherConsensus",
    "TeacherConsensusEngine",
    "VerifiableOracle",
]
