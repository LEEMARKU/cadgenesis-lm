"""tests/continual_learning/test_ewc.py
=====================================
Unit tests for EWC regularization.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cadgenesis.continual_learning.ewc import EWC


def _make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _batches(
    n: int = 8,
    batch_size: int = 16,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(batch_size, 4, generator=generator),
            torch.randint(0, 2, (batch_size,), generator=generator),
        )
        for _ in range(n)
    ]


def test_penalty_zero_at_anchor():
    model = _make_model()
    ewc = EWC(lambda_=100.0)
    ewc.register_fisher(model, _batches(), nn.CrossEntropyLoss(), n_samples=4)
    assert ewc.penalty(model).item() == 0.0


def test_penalty_grows_with_perturbation():
    model = _make_model()
    ewc = EWC(lambda_=100.0)
    ewc.register_fisher(model, _batches(), nn.CrossEntropyLoss(), n_samples=4)
    with torch.no_grad():
        model[0].weight.add_(0.01)
    first = ewc.penalty(model).item()
    assert first > 0.0
    with torch.no_grad():
        model[0].weight.add_(0.1)
    second = ewc.penalty(model).item()
    assert second > first


def test_penalty_scales_with_lambda():
    model = _make_model()
    ewc_small = EWC(lambda_=10.0)
    ewc_small.register_fisher(model, _batches(), nn.CrossEntropyLoss(), n_samples=4)
    ewc_large = EWC(lambda_=1000.0)
    ewc_large.register_fisher(model, _batches(), nn.CrossEntropyLoss(), n_samples=4)
    with torch.no_grad():
        model[0].weight.add_(0.05)
    small = ewc_small.penalty(model).item()
    large = ewc_large.penalty(model).item()
    assert large > small


def test_ewc_loss_alias():
    model = _make_model()
    ewc = EWC(lambda_=1.0)
    ewc.register_fisher(model, _batches(n=2), nn.CrossEntropyLoss(), n_samples=2)
    with torch.no_grad():
        model[2].bias.add_(0.2)
    assert ewc.ewc_loss(model).item() == pytest.approx(ewc.penalty(model).item())


def test_none_grads_guarded():
    class PartialModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Parameter(torch.randn(2))
            self.unused = nn.Parameter(torch.randn(2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.used

    model = PartialModel()
    ewc = EWC(lambda_=1.0)
    batches = [(torch.randn(8, 2), torch.randint(0, 2, (8,)))]
    ewc.register_fisher(model, batches, nn.CrossEntropyLoss(), n_samples=1)
    assert ewc.penalty(model).item() == 0.0
    assert "unused" not in ewc.fishers


def test_register_fisher_empty_raises():
    ewc = EWC()
    with pytest.raises(ValueError):
        ewc.register_fisher(_make_model(), [], nn.CrossEntropyLoss(), n_samples=5)


def test_reset():
    model = _make_model()
    ewc = EWC(lambda_=1.0)
    ewc.register_fisher(model, _batches(n=2), nn.CrossEntropyLoss(), n_samples=2)
    ewc.reset()
    assert ewc.penalty(model).item() == 0.0
    assert ewc.fishers == {}
