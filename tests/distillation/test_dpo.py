"""tests/distillation/test_dpo.py"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from cadgenesis.distillation.dpo import DPOTrainer

VOCAB = 20
DIM = 16


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, DIM)
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(ids))


def make_logprob_fn() -> callable:
    def logprob_fn(
        model: nn.Module, prompt_ids: list[int], completion_ids: list[int]
    ) -> torch.Tensor:
        seq = torch.tensor(prompt_ids + completion_ids, dtype=torch.long)
        logits = model(seq)
        logprobs = torch.log_softmax(logits, dim=-1)
        start = len(prompt_ids)
        return logprobs[start:].gather(1, seq[start:].unsqueeze(1)).squeeze(1).sum()

    return logprob_fn


def make_trainer(label_smoothing: float = 0.0, lr: float = 1e-5) -> DPOTrainer:
    torch.manual_seed(0)
    policy = TinyPolicy()
    ref_model = copy.deepcopy(policy)
    return DPOTrainer(
        policy,
        ref_model,
        make_logprob_fn(),
        lr=lr,
        beta=0.1,
        label_smoothing=label_smoothing,
    )


def test_dpo_loss_is_log_two_for_identical_models():
    trainer = make_trainer()
    loss, stats = trainer.dpo_loss([5, 5], [0, 1, 2, 3], [0, 9, 9, 9])
    assert loss.item() == pytest.approx(math.log(2), abs=1e-2)
    assert stats["dpo"] == pytest.approx(math.log(2), abs=1e-2)


def test_dpo_training_decreases_loss():
    trainer = make_trainer(lr=1e-2)
    prompt = [5, 5]
    chosen = [0, 1, 2, 3]
    rejected = [0, 9, 9, 9]
    before = trainer.dpo_loss(prompt, chosen, rejected)[0].item()
    for _ in range(20):
        trainer.train_step([prompt], [chosen], [rejected])
    after = trainer.dpo_loss(prompt, chosen, rejected)[0].item()
    assert after < before


def test_train_step_accumulates_gradients():
    trainer = make_trainer()
    trainer.train_step([[5, 5]], [[0, 1, 2, 3]], [[0, 9, 9, 9]])
    assert trainer.policy.embedding.weight.grad is not None


def test_train_step_requires_matching_list_lengths():
    trainer = make_trainer()
    with pytest.raises(ValueError):
        trainer.train_step([[5, 5]], [[0, 1, 2, 3]], [[0, 9, 9, 9], [0, 8, 8, 8]])


def test_label_smoothing_changes_loss():
    torch.manual_seed(0)
    ref_model = TinyPolicy()
    logprob_fn = make_logprob_fn()
    trainer_plain = DPOTrainer(TinyPolicy(), ref_model, logprob_fn, label_smoothing=0.0)
    trainer_smooth = DPOTrainer(TinyPolicy(), ref_model, logprob_fn, label_smoothing=0.1)
    prompt, chosen, rejected = [5, 5], [0, 1, 2, 3], [0, 9, 9, 9]
    loss_plain, _ = trainer_plain.dpo_loss(prompt, chosen, rejected)
    loss_smooth, _ = trainer_smooth.dpo_loss(prompt, chosen, rejected)
    assert torch.isfinite(loss_plain)
    assert torch.isfinite(loss_smooth)
    assert abs(loss_plain.item() - loss_smooth.item()) > 1e-6
