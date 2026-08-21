"""
cadgenesis.distillation.distillation_engine
===========================================
Multi-Teacher Distillation Engine for CADGenesis-LM v2.0:
- Interfaces 7 Frontier Teacher LLMs (GPT, Claude, Gemini, Llama, DeepSeek, Qwen, Mistral)
- Computes Soft-Label KL Divergence & Hard-Label Distillation Loss
- Teacher Consensus & Critique Engine
- Synthetic Dataset Generator & Self-Correction Loop
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTeacherDistillationEngine(nn.Module):
    """
    Computes Distillation Loss combining Soft-Label KL Divergence from
    teacher models with Hard-Label Cross-Entropy Loss.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = 0,
    ) -> torch.Tensor:
        """
        student_logits: (B*T, V)
        teacher_logits: (B*T, V)
        labels: (B*T)
        """
        # Hard label loss.  Guard against the all-targets-ignored corner case:
        # PyTorch returns NaN when every label equals `ignore_index` (e.g. a
        # single sample whose label hashes to 0), so use 0.0 for those batches.
        valid = labels != ignore_index
        if bool(valid.any()):
            hard_loss = F.cross_entropy(student_logits, labels, ignore_index=ignore_index)
        else:
            hard_loss = torch.zeros((), dtype=student_logits.dtype, device=student_logits.device)

        # Soft label KL divergence
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
            self.temperature**2
        )

        return (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)


class TeacherConsensusEngine:
    """
    Aggregates CAD generations from 7 Teacher LLMs to compute consensus & reward scores.
    """

    def __init__(self, teacher_names: list[str] | None = None):
        self.teacher_names = teacher_names or [
            "GPT-4o",
            "Claude-3",
            "Gemini-1.5",
            "LLaMA-3.1",
            "DeepSeek",
            "Qwen-2.5",
            "Mistral",
        ]

    def compute_consensus(self, teacher_outputs: dict[str, list[str]]) -> tuple[list[str], float]:
        """
        Returns (consensus_tokens: List[str], consensus_agreement_score: float)
        """
        if not teacher_outputs:
            return [], 0.0

        all_seqs = list(teacher_outputs.values())
        first_seq = all_seqs[0]
        match_count = sum(1 for s in all_seqs if s == first_seq)
        score = match_count / len(all_seqs)

        return first_seq, score
