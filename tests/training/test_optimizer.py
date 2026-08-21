"""tests/training/test_optimizer.py"""

from __future__ import annotations

import pytest

from cadgenesis.training.optimizer import OPTIMIZERS, build_optimizer, lora_param_groups

torch = pytest.importorskip("torch")


class TinyLoraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(4, 4)
        self.base.weight.requires_grad = False
        self.base.bias.requires_grad = False
        self.lora_a = torch.nn.Parameter(torch.randn(4, 2))
        self.lora_b = torch.nn.Parameter(torch.randn(2, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + x @ self.lora_a @ self.lora_b


def test_optimizers_available():
    assert "adamw" in OPTIMIZERS
    assert "adam" in OPTIMIZERS


def test_lora_param_groups_splits_params():
    model = TinyLoraModel()
    groups = lora_param_groups(model, lr=1e-4, base_lr=1e-5, weight_decay=0.01)
    assert len(groups) == 1
    lora_group = groups[0]
    assert lora_group["lr"] == 1e-4
    assert len(lora_group["params"]) == 2


def test_lora_param_groups_excludes_frozen():
    model = TinyLoraModel()
    groups = lora_param_groups(model)
    total = sum(len(g["params"]) for g in groups)
    assert total == 2  # only the LoRA parameters are trainable


def test_lora_param_groups_no_trainable_raises():
    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad = False
    model.bias.requires_grad = False
    with pytest.raises(ValueError):
        lora_param_groups(model)


def test_build_optimizer_adamw():
    model = TinyLoraModel()
    optimizer = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.1)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1


def test_build_optimizer_sgd():
    model = TinyLoraModel()
    optimizer = build_optimizer(model, "sgd", lr=0.01, momentum=0.9)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9


def test_build_optimizer_lora_only():
    model = TinyLoraModel()
    optimizer = build_optimizer(model, "adamw", lr=1e-4, lora_only=True)
    assert len(optimizer.param_groups) == 1
    total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    assert total == 4 * 2 + 2 * 4


def test_build_optimizer_unknown_type_raises():
    model = TinyLoraModel()
    with pytest.raises(ValueError):
        build_optimizer(model, "not-an-optimizer")


def test_build_optimizer_no_trainable_raises():
    model = torch.nn.Linear(4, 4)
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError):
        build_optimizer(model)
