"""
cadgenesis.quantization.bitnet
==============================
BitNet b1.58 — 1.58-bit ternary weights for CPU-friendly LLM inference.

Motivation
----------
Multipliers dominate LLM cost.  BitNet b1.58 (Wang et al., 2024) constrains
every weight to {-1, 0, +1} * (per-tensor scale): a matmul becomes a sequence
of additions, which is ~10x cheaper on GPU and even more decisive on CPU-only
deployments.  Activations are quantised per-token to int8 with a
per-token ``alpha = max(|x|)/127`` scale.

``BitLinear`` keeps full-precision parameters and a straight-through estimator
so the model is *trained* in fp32 while its forward computation is quantised —
BitNet's core recipe.  At inference the path is deterministic and bit-exact.

:func:`apply_bitnet` rewrites a model's ``nn.Linear`` modules in place.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """
    nn.Linear with BitNet b1.58 quantisation (ternary weights, int8 activations).

    Uses a straight-through estimator: the forward pass runs on the quantised
    tensors, but gradients flow to the full-precision parameters, so the model
    can be trained end-to-end and then served with zero extra quantisation work.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    @staticmethod
    def _ternarize(w: torch.Tensor) -> torch.Tensor:
        """w -> {-scale, 0, +scale} with scale = mean(|w|) (b1.58)."""
        scale = w.abs().mean().clamp_min(1e-9)
        return (w / scale).round().clamp(-1.0, 1.0) * scale

    @staticmethod
    def _quantize_act(x: torch.Tensor) -> torch.Tensor:
        """Per-token int8 activation quantisation: round(x/alpha)*alpha."""
        alpha = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / 127.0
        return torch.clamp(torch.round(x / alpha), -127, 127) * alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w_tern = self._ternarize(w)
        x_q = self._quantize_act(x)
        # Straight-through estimator: forward on quantised, gradient to fp.
        w_ste = w + (w_tern - w).detach()
        x_ste = x + (x_q - x).detach()
        return F.linear(x_ste, w_ste, self.bias)


def apply_bitnet(
    model: nn.Module,
    exclude: tuple[str, ...] = (),
    verbose: bool = False,
) -> int:
    """
    Replace every ``nn.Linear`` in ``model`` with :class:`BitLinear` in place,
    preserving weight/bias tensors.  Layers whose weights are *shared* (weight
    tying, e.g. the LM head tied to the input embedding) and modules whose names
    match ``exclude`` are left untouched.  Returns the number of layers swapped.
    """
    swapped = 0
    # Weight tying: count every module-level reference to a parameter's storage
    # (``model.parameters()`` dedupes shared tensors, so traverse the module
    # tree instead).  Storage referenced by >1 module is left untouched (e.g.
    # the LM head tied to the input embedding).
    ptr_counts: dict[int, int] = {}
    for _, module in model.named_modules():
        for param in module.parameters(recurse=False):
            ptr_counts[param.data_ptr()] = ptr_counts.get(param.data_ptr(), 0) + 1
    tied_ptrs = {ptr for ptr, n in ptr_counts.items() if n > 1}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, BitLinear):
            continue
        if any(ex in name for ex in exclude):
            continue
        if module.weight.data_ptr() in tied_ptrs:
            continue

        bit = BitLinear(module.in_features, module.out_features, module.bias is not None)
        bit.weight.data = module.weight.data
        if module.bias is not None:
            bit.bias.data = module.bias.data
        bit.to(module.weight.device)

        # Splice into the parent's ModuleList/children slot.
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        child_name = name.split(".")[-1]
        if isinstance(parent, nn.ModuleList):
            parent[int(child_name)] = bit
        elif isinstance(parent, nn.ModuleDict):
            parent[child_name] = bit
        else:
            setattr(parent, child_name, bit)
        swapped += 1

    if verbose:
        print(f"[apply_bitnet] swapped {swapped} linear layers.")
    return swapped
