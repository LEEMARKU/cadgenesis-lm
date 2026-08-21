"""tests/continual_learning/test_adapter_isolation.py
====================================================
Unit tests for task-isolated adapters.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cadgenesis.continual_learning.adapter_isolation import TaskAdapterRegistry


class AdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_a = nn.Linear(4, 2)
        self.lora_b = nn.Linear(2, 4)
        self.adapter_scale = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.adapter_scale


def _parameter_names(model: nn.Module) -> set[str]:
    return {name for name, _ in model.named_parameters()}


def test_register_task_matches_key_substrings():
    model = AdapterModel()
    registry = TaskAdapterRegistry()
    task = registry.register_task("t1", model, ["lora", "adapter"])
    assert task.task_id == "t1"
    assert set(task.param_names) == {
        "lora_a.weight",
        "lora_a.bias",
        "lora_b.weight",
        "lora_b.bias",
        "adapter_scale",
    }
    assert "base.weight" in task.frozen
    assert "base.bias" in task.frozen
    assert set(task.frozen) == _parameter_names(model) - set(task.param_names)


def test_isolate_freezes_correct_params_and_release_restores():
    model = AdapterModel()
    registry = TaskAdapterRegistry()
    task = registry.register_task("t1", model, ["lora"])
    registry.isolate("t1", model)
    for name, param in model.named_parameters():
        assert param.requires_grad == (name in task.param_names)
    assert registry.is_isolated("t1")
    registry.release("t1", model)
    assert not registry.is_isolated("t1")
    assert all(param.requires_grad for _, param in model.named_parameters())


def test_release_restores_custom_requires_grad_flags():
    model = AdapterModel()
    registry = TaskAdapterRegistry()
    registry.register_task("t1", model, ["lora"])
    with torch.no_grad():
        model.base.bias.fill_(0.0)
    model.base.bias.requires_grad_(False)
    registry.isolate("t1", model)
    registry.release("t1", model)
    assert not model.base.bias.requires_grad  # pre-isolation flag preserved
    assert model.lora_a.weight.requires_grad


def test_isolate_unregistered_raises():
    model = AdapterModel()
    registry = TaskAdapterRegistry()
    with pytest.raises(ValueError):
        registry.isolate("missing", model)
    with pytest.raises(ValueError):
        registry.release("missing", model)


def test_duplicate_registration_raises():
    model = AdapterModel()
    registry = TaskAdapterRegistry()
    registry.register_task("t1", model, ["lora"])
    with pytest.raises(ValueError):
        registry.register_task("t1", model, ["adapter"])


def test_nested_isolations_restore_their_own_baseline():
    model = AdapterModel()
    registry = TaskAdapterRegistry()
    registry.register_task("lora_task", model, ["lora"])
    registry.register_task("norm_task", model, ["adapter"])
    registry.isolate("lora_task", model)
    registry.isolate("norm_task", model)
    assert model.adapter_scale.requires_grad
    assert not model.lora_a.weight.requires_grad
    assert not model.base.weight.requires_grad
    registry.release("norm_task", model)
    # lora_task is still active, so its mask is re-applied after restoring
    # norm_task's baseline: lora trainable, base + adapter_scale frozen
    assert not model.adapter_scale.requires_grad
    assert model.lora_a.weight.requires_grad
    assert not model.base.weight.requires_grad
    registry.release("lora_task", model)
    assert all(param.requires_grad for _, param in model.named_parameters())
