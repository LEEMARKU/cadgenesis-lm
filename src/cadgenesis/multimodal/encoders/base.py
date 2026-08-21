"""cadgenesis.multimodal.encoders.base
====================================
Base contract for every modality encoder (Pillar 3).

A :class:`MultimodalEncoder` turns one modality's raw input into a tensor of
raw features of shape ``(B, feature_dim)`` (or ``(B, T, feature_dim)`` for
sequence-aware modalities).  The shared projection head in the
:class:`~cadgenesis.multimodal.embeddings.SharedEngineeringEmbeddingSpace`
projects those raw features into the common latent space — encoders never
know about the shared embedding dimension.

Two entry points are provided:

* :meth:`forward` — the ``nn.Module`` tensor contract, batched tensors in /
  batched tensors out.  Used inside fused pipelines and for gradient flow.
* :meth:`encode` — the ergonomic contract that accepts *structured* inputs
  (lists of strings, CAD documents, sketch documents, ...) and returns a
  batched tensor of raw features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality


class MultimodalEncoder(nn.Module, ABC):
    """Abstract base class for all modality encoders."""

    modality: ClassVar[Modality]
    #: True when the encoder can emit a sequence of tokens (T > 1).
    sequence_aware: ClassVar[bool] = False

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        #: Raw feature dimension produced by the encoder (fed to the shared space).
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tensor contract: ``(B, ...) -> (B, feature_dim)`` or
        ``(B, T, feature_dim)`` for sequence-aware encoders."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        """Structured contract: any supported input -> batched raw features."""
        raise NotImplementedError

    def to_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward` used by the multimodal facade."""
        return self.forward(x)


def tensorize(data: object, dtype: str | torch.dtype | None = None) -> torch.Tensor:
    """Convert a numpy array / nested list / torch tensor into a torch tensor.

    ``dtype`` optionally casts the result (e.g. ``"float32"``).  Raises
    ``TypeError`` for unsupported inputs so encoders fail fast with a clear
    message instead of a confusing downstream error.
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        import numpy as np

        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            try:
                tensor = torch.tensor(data)
            except (TypeError, ValueError) as exc:
                raise TypeError("nested lists must contain homogeneous numeric values") from exc
        else:
            raise TypeError(
                f"cannot tensorize object of type {type(data).__name__}; expected "
                "torch.Tensor, numpy.ndarray or a nested list of numbers"
            )
    if dtype is not None:
        if isinstance(dtype, str):
            resolved: torch.dtype = {
                "float32": torch.float32,
                "float64": torch.float64,
                "long": torch.long,
                "int64": torch.long,
            }.get(dtype, torch.float32)
        else:
            resolved = dtype
        tensor = tensor.to(resolved)
    return tensor


__all__ = ["MultimodalEncoder", "tensorize"]
