"""tests/continual_learning/test_continual_trainer.py
====================================================
Unit tests for the continual training loop.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cadgenesis.continual_learning.continual_trainer import ContinualTrainer
from cadgenesis.continual_learning.evaluator import ContinualEvaluator
from cadgenesis.continual_learning.ewc import EWC


def _make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))


def _make_task_data(
    seed: int, invert: bool = False, n: int = 100, dim: int = 4
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=generator)
    labels = (x[:, 0] > 0).long()
    if invert:
        labels = 1 - labels
    return list(zip(x.split(16), labels.split(16), strict=True))


def test_train_task_reduces_loss_and_learns():
    model = _make_model()
    trainer = ContinualTrainer(model, device="cpu")
    batches = _make_task_data(seed=0)
    result = trainer.train_task("task_a", batches, epochs=3)
    assert trainer.tasks == ["task_a"]
    assert len(result["loss_steps"]) == 3 * len(batches)
    assert result["loss_steps"][0] > result["loss_steps"][-1]
    assert result["final_acc"] > 0.8


def test_regularizer_is_added_to_loss():
    model = _make_model()
    ewc = EWC(lambda_=50.0)
    ewc.register_fisher(model, _make_task_data(seed=1), nn.CrossEntropyLoss(), n_samples=4)
    trainer = ContinualTrainer(model, regularizers=[ewc.penalty], device="cpu")
    result = trainer.train_task("task_a", _make_task_data(seed=0), epochs=2)
    assert result["loss_steps"][0] > result["loss_steps"][-1]
    assert result["final_acc"] > 0.5


def test_custom_optimizer_factory():
    model = _make_model()
    trainer = ContinualTrainer(
        model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        device="cpu",
    )
    result = trainer.train_task("task_a", _make_task_data(seed=2), epochs=2)
    assert result["final_acc"] > 0.6


def test_fit_and_evaluate_records_accuracy():
    model = _make_model()
    trainer = ContinualTrainer(model, device="cpu")
    evaluator = ContinualEvaluator()
    result = trainer.fit_and_evaluate(
        "task_a", _make_task_data(seed=0), _make_task_data(seed=7), evaluator, epochs=2
    )
    assert result["eval_acc"] > 0.8
    assert evaluator.accuracy_history("task_a") == [pytest.approx(result["eval_acc"])]
    assert evaluator.forgetting("task_a") == 0.0


def test_forgetting_measurable_across_two_tasks():
    model = _make_model()
    trainer = ContinualTrainer(model, device="cpu")
    evaluator = ContinualEvaluator()
    eval_a = _make_task_data(seed=7, n=300)
    trainer.fit_and_evaluate("task_a", _make_task_data(seed=0, n=300), eval_a, evaluator, epochs=3)
    first_acc = evaluator.accuracy_history("task_a")[-1]
    assert first_acc > 0.8
    trainer.fit_and_evaluate(
        "task_b", _make_task_data(seed=0, invert=True, n=300), eval_a, evaluator, epochs=4
    )
    second_acc = trainer.evaluate(eval_a)
    evaluator.record_task_acc("task_a", second_acc)
    assert second_acc < first_acc
    assert evaluator.forgetting("task_a") > 0.2  # inverse task provokes forgetting
    assert evaluator.forgetting("task_b") == 0.0
