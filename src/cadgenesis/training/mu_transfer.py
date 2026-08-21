"""
cadgenesis.training.mu_transfer
===============================
µTransfer — maximal update parametrization (Yang et al., 2022).

Why it matters
--------------
Hyperparameters found on a small model do *not* transfer to a large one under
standard (Pytorch-default) parametrisation: as width grows, the optimal LR and
init scales drift.  µP fixes the parametrisation so that the *feature-learning
rate* is width-independent — an LR tuned on ``nano`` then transfers to
``large`` unchanged.  This is the standard method for turning a scale ladder
(such as :meth:`CADConfig.from_preset`) into a credible scaling story.

µP rules implemented here
-------------------------
1. Embedding output scaling: multiply the *input* embeddings (language +
   CAD) by ``alpha / sqrt(d)`` (default ``alpha = 1``) so the hidden norm does
   not grow with width.
2. Output / readout init: the final logits projection is initialised with
   std ``1 / d`` (vs ``1/sqrt(d)`` default) — its LR is boosted instead.
3. LR multipliers per parameter kind, all relative to a base LR:
       readout (final logits head)  x d   (boosted — it was shrunk at init)
       embeddings / attention / FFN  x1
       biases                       x1, no weight decay
   (bias params are excluded from the learning-rate scaling and decay.)

:func:`mu_param_groups` returns ``[{params, lr, weight_decay}, ...]`` ready for
``torch.optim.AdamW``.  ``lr_multiplier`` can be overridden per module name.

Note on weight tying: the LM head is tied to ``cad_embed`` in this model, so
``named_parameters()`` reports the shared tensor only once (under the embedding
name).  Param groups therefore key modules by parameter *data pointer* so the
tied head still receives the boosted readout LR.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn

EMBED_ALPHA = 1.0

# Attribute names of the model's *input* token embeddings (the decoder CAD
# embedding is also the tied LM head — we only scale its *input* usage, which
# is exactly the µP embedding rule).
_INPUT_EMBED_NAMES = ("lang_embed", "cad_embed")


def _readout_names(model: nn.Module) -> set[str]:
    """Name of the single final logits projection (the LM head)."""
    names: set[str] = set()
    for name, module in model.named_modules():
        # Only the *top-level* head, never the per-block attention outputs.
        if name == "out_proj" and isinstance(module, nn.Linear):
            names.add(name)
    return names


def _param_module_names(model: nn.Module) -> dict[int, list[str]]:
    """
    Map ``param.data_ptr() -> [owning module names]``.

    ``named_parameters()`` dedupes tied weights, so a head tied to the input
    embedding only shows up under the embedding name.  Tracking data pointers
    (shared by tied tensors) recovers every owning module.
    """
    mapping: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            mapping.setdefault(param.data_ptr(), []).append(name)
    return mapping


def _wrap_input_embedding(model: nn.Module, d_model: int) -> None:
    """Scale input-embedding outputs by ``EMBED_ALPHA / sqrt(d)`` at forward
    time so gradients remain exact."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Embedding):
            continue
        if name not in _INPUT_EMBED_NAMES:
            continue
        original_forward = module.forward

        def scaled_forward(input_: torch.Tensor, *args, _original=original_forward, **kwargs):
            out = _original(input_, *args, **kwargs)
            return out * (EMBED_ALPHA / d_model**0.5)

        module.forward = cast(Callable[..., Any], scaled_forward)  # type: ignore[method-assign]


def apply_mu_transfer(model: nn.Module, d_model: int) -> nn.Module:
    """
    Apply µP init/scaling in place.

    * Input embeddings (language + CAD) get output scaling ``alpha = 1/sqrt(d)``.
    * The readout (final logits projection) is rescaled from the model's
      ``1/√d``-class init down to std ``1/d`` (its LR is boosted separately).

    The embedding parameters themselves are not re-initialised (their LR is
    set by :func:`mu_param_groups`).
    """
    readout = _readout_names(model)

    for name, module in model.named_modules():
        if name in readout and isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.mul_(d_model**-0.5)  # 1/sqrt(d) -> 1/d

    _wrap_input_embedding(model, d_model)
    return model


def mu_param_groups(
    model: nn.Module,
    base_lr: float,
    d_model: int,
    weight_decay: float = 0.1,
) -> list[dict]:
    """
    Build ``torch.optim.AdamW`` param groups with µP LR multipliers.

    * readout (final logits projection, incl. a tied head): ``base_lr * d``
    * everything else (embeddings, attention, FFN): ``base_lr``
    * biases: ``base_lr`` with no weight decay (excluded from decay).
    """
    readout = _readout_names(model)
    module_names = _param_module_names(model)
    groups: list[dict] = [
        {"params": [], "lr": base_lr * d_model},
        {"params": [], "lr": base_lr},
        {"params": [], "lr": base_lr, "weight_decay": 0.0},
    ]
    for _, param in model.named_parameters():
        owners = module_names.get(param.data_ptr(), [])
        if param.ndim == 1:  # biases / norms
            groups[2]["params"].append(param)
        elif any(owner in readout for owner in owners):
            groups[0]["params"].append(param)
        else:
            groups[1]["params"].append(param)
    groups[0]["weight_decay"] = weight_decay
    groups[1]["weight_decay"] = weight_decay
    return groups


def build_mu_optimizer(
    model: nn.Module,
    base_lr: float,
    d_model: int,
    weight_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """Convenience: :class:`AdamW` configured with µP LR groups."""
    return torch.optim.AdamW(mu_param_groups(model, base_lr, d_model, weight_decay))
