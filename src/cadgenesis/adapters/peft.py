"""cadgenesis.adapters.peft
=========================
PEFT framework facade.

Thin, typed facade over :mod:`cadgenesis.adapters.lora` providing attach /
activate / deactivate / merge / list semantics. Multiple adapters may be
attached to one model; only the active adapter's LoRA delta is wired into the
module tree (they share the same base weights), and ``merge`` folds a delta
into the base weights.

Note: attaching adapters with *different* ``target_layers`` sets to the same
model is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cadgenesis.adapters.lora import LoRALinear

DEFAULT_TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "out_proj"]


@dataclass
class _LoRAInstallation:
    """Where a LoRA wrapper was installed and what it replaced."""

    parent: nn.Module
    attr_name: str
    original: nn.Linear
    wrapper: LoRALinear
    weight_requires_grad: bool
    bias_requires_grad: bool


class PEFTAdapter:
    """Facade for attaching, activating, merging and listing LoRA adapters."""

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._active: str | None = None
        self._attached: dict[str, list[_LoRAInstallation]] = {}

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def active(self) -> str | None:
        return self._active

    def attach(
        self,
        model: nn.Module,
        adapter_id: str,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_layers: list[str] | None = None,
    ) -> nn.Module:
        """Install LoRA modules on matching Linear layers; returns ``model``."""
        if adapter_id in self._attached:
            raise ValueError(f"adapter {adapter_id!r} is already attached")
        if self._model is None:
            self._model = model
        elif self._model is not model:
            raise ValueError("PEFTAdapter is already bound to a different model")

        targets = target_layers if target_layers is not None else DEFAULT_TARGET_LAYERS
        installations: list[_LoRAInstallation] = []
        for _name, module in model.named_modules():
            for attr_name, child in module.named_children():
                if not any(target in attr_name for target in targets):
                    continue
                base = _resolve_base_linear(child)
                if base is None:
                    continue
                wrapper = LoRALinear(base, rank=r, alpha=alpha, dropout=dropout)
                setattr(module, attr_name, wrapper)
                installations.append(
                    _LoRAInstallation(
                        parent=module,
                        attr_name=attr_name,
                        original=base,
                        wrapper=wrapper,
                        weight_requires_grad=base.weight.requires_grad,
                        bias_requires_grad=base.bias is not None and base.bias.requires_grad,
                    )
                )
        if not installations:
            raise ValueError(f"no Linear layers matched target_layers={targets!r}")
        self._attached[adapter_id] = installations
        self._active = adapter_id
        return model

    def activate(self, adapter_id: str) -> None:
        """Wire the adapter's LoRA wrappers into the module tree."""
        installations = self._attached.get(adapter_id)
        if installations is None:
            raise ValueError(f"adapter {adapter_id!r} is not attached")
        for installation in installations:
            setattr(installation.parent, installation.attr_name, installation.wrapper)
        self._active = adapter_id

    def deactivate(self) -> None:
        """Restore all base Linears so forward passes use unadapted weights."""
        restored: set[tuple[int, str]] = set()
        for installations in self._attached.values():
            for installation in installations:
                key = (id(installation.parent), installation.attr_name)
                if key in restored:
                    continue
                restored.add(key)
                setattr(installation.parent, installation.attr_name, installation.original)
        self._active = None

    def list_adapters(self) -> list[str]:
        """Attached adapter ids, in attach order."""
        return list(self._attached)

    def merge(self, adapter_id: str) -> nn.Module:
        """Fold the adapter's LoRA delta into base weights; returns the model."""
        model = self._model
        if model is None:
            raise ValueError("no model is bound to this PEFTAdapter")
        installations = self._attached.get(adapter_id)
        if installations is None:
            raise ValueError(f"adapter {adapter_id!r} is not attached")
        for installation in installations:
            wrapper = installation.wrapper
            delta = (wrapper.lora_B @ wrapper.lora_A) * wrapper.scaling
            with torch.no_grad():
                installation.original.weight.add_(delta)
            self._restore(installation)
        del self._attached[adapter_id]
        if self._active == adapter_id:
            self._active = None
        return model

    def _restore(self, installation: _LoRAInstallation) -> None:
        setattr(installation.parent, installation.attr_name, installation.original)
        installation.original.weight.requires_grad = installation.weight_requires_grad
        if installation.original.bias is not None:
            installation.original.bias.requires_grad = installation.bias_requires_grad


def _resolve_base_linear(child: nn.Module) -> nn.Linear | None:
    """The base nn.Linear behind either a Linear or an installed LoRA wrapper."""
    if isinstance(child, nn.Linear):
        return child
    if isinstance(child, LoRALinear) and isinstance(child.original_linear, nn.Linear):
        return child.original_linear
    return None
