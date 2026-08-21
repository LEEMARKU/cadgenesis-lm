"""
cadgenesis.transformer.self_designing.pruning
=======================================
Layer Pruning for the Self-Designing Transformer.

``LayerPruningController`` identifies the least-important layers of the
backbone and *logically* prunes them: pruned layers are forced to a zero
layer-gate (exact skip) without destroying their weights, so pruning is
reversible (``unprune``).  Importance is estimated with a lightweight,
gradient-free proxy — the mean magnitude of each layer's weight tensors —
which correlates well with utility in trained networks.

Algorithm
---------
    importance(layer) = mean(|W|) over all parameter tensors of the block

    prune(fraction):
        rank layers ascending by importance
        mark the lowest ``fraction`` as pruned
        → layer_gate == 0 for those layers (exact skip)

Complexity
----------
    importance sweep: O(P)  where P = total parameters
    prune/unprune:    O(L)  with L = number of layers
"""

from __future__ import annotations

import torch.nn as nn

from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


class LayerPruningController:
    """
    Tracks per-layer importance and reversible pruning state.

    ``encoder`` layers are indexed ``0 .. E-1``, ``decoder`` layers
    ``0 .. D-1``.
    """

    def __init__(self, num_encoder_layers: int, num_decoder_layers: int):
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.importance: dict[str, list[float]] = {
            "encoder": [0.0] * num_encoder_layers,
            "decoder": [0.0] * num_decoder_layers,
        }
        self.pruned: dict[str, list[bool]] = {
            "encoder": [False] * num_encoder_layers,
            "decoder": [False] * num_decoder_layers,
        }

    # ------------------------------------------------------------- importance

    def record_importance(self, layer_type: str, index: int, score: float) -> None:
        """Store an externally computed importance score for a layer."""
        self._check(layer_type, index)
        self.importance[layer_type][index] = float(score)

    def compute_importance(self, model: GeometryAwareTransformer) -> dict[str, list[float]]:
        """
        Gradient-free structure importance: mean |W| over each block's params.

        ``model`` is a ``GeometryAwareTransformer`` exposing
        ``encoder_blocks`` / ``decoder_blocks`` ModuleLists.  The controller
        resizes its tracking arrays to the model's actual layer counts.
        """
        self._resize(len(model.encoder_blocks), len(model.decoder_blocks))
        for key, blocks in (("encoder", model.encoder_blocks), ("decoder", model.decoder_blocks)):
            for i, block in enumerate(blocks):
                self.importance[key][i] = self._block_importance(block)
        return self.importance

    def _resize(self, num_encoder_layers: int, num_decoder_layers: int) -> None:
        """Resize tracking lists to match a new architecture."""
        sizes = {"encoder": num_encoder_layers, "decoder": num_decoder_layers}
        for key in ("encoder", "decoder"):
            target = sizes[key]
            if len(self.importance[key]) != target:
                self.importance[key] = [0.0] * target
                self.pruned[key] = [False] * target
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

    @staticmethod
    def _block_importance(block: nn.Module) -> float:
        total, count = 0.0, 0
        for p in block.parameters():
            total += p.detach().abs().mean().item()
            count += 1
        return total / max(1, count)

    # ---------------------------------------------------------------- pruning

    def prune_layers(
        self,
        fraction: float = 0.25,
        layer_type: str | None = None,
    ) -> list[tuple[str, int]]:
        """
        Mark the lowest-importance layers as pruned.

        Returns the list of pruned (layer_type, index) pairs.
        """
        targets = ["encoder", "decoder"] if layer_type is None else [layer_type]
        pruned_now: list[tuple[str, int]] = []
        for key in targets:
            self._check_type(key)
            n = len(self.importance[key])
            to_prune = max(1, round(n * fraction)) if n > 1 else (1 if n else 0)
            ranked = sorted(range(n), key=lambda i: (self.importance[key][i], i))
            for idx in ranked[:to_prune]:
                if not self.pruned[key][idx]:
                    self.pruned[key][idx] = True
                    pruned_now.append((key, idx))
        return pruned_now

    def unprune(self, layer_type: str, index: int) -> None:
        self._check(layer_type, index)
        self.pruned[layer_type][index] = False

    def unprune_all(self) -> None:
        for key in self.pruned:
            self.pruned[key] = [False] * len(self.pruned[key])

    def is_pruned(self, layer_type: str, index: int) -> bool:
        self._check(layer_type, index)
        return self.pruned[layer_type][index]

    def pruned_list(self) -> list[tuple[str, int]]:
        return [
            (key, idx)
            for key in ("encoder", "decoder")
            for idx, p in enumerate(self.pruned[key])
            if p
        ]

    def effective_layers(self) -> tuple[int, int]:
        """(active_encoder_layers, active_decoder_layers)."""
        return (
            self.num_encoder_layers - sum(self.pruned["encoder"]),
            self.num_decoder_layers - sum(self.pruned["decoder"]),
        )

    # -------------------------------------------------------------- helpers

    def _check(self, layer_type: str, index: int) -> None:
        self._check_type(layer_type)
        n = len(self.pruned[layer_type])
        if not (0 <= index < n):
            raise IndexError(f"{layer_type} layer {index} out of range [0, {n}).")

    @staticmethod
    def _check_type(layer_type: str) -> None:
        if layer_type not in ("encoder", "decoder"):
            raise ValueError(f"layer_type must be 'encoder' or 'decoder'; got {layer_type!r}.")

    def report(self) -> dict:
        return {
            "pruned": self.pruned_list(),
            "effective_layers": self.effective_layers(),
            "importance": {k: [round(v, 4) for v in vs] for k, vs in self.importance.items()},
        }
