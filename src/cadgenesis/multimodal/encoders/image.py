"""cadgenesis.multimodal.encoders.image
=====================================
Image encoder (raster CAD previews, renders, screenshots).

Images are normalised to a fixed ``(3, height, width)`` tensor via
``tensorize`` and fed through a small convolutional backbone followed by a
projection MLP into the shared raw feature space.  The backbone is trained
as part of the shared engineering embedding space (contrastive / metric
objectives), and can be swapped for a larger visual tower.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize


@dataclass
class ImageDocument:
    """Wrapped image tensor plus optional metadata."""

    data: Any
    width: int = 0
    height: int = 0
    source: str = "raster"

    def __post_init__(self) -> None:
        tensor = tensorize(self.data, dtype="float32")
        if tensor.dim() == 3:
            channels, height, width = tensor.shape
        elif tensor.dim() == 2:
            channels, height, width = 1, *tensor.shape
        else:
            raise ValueError(f"image data must be (C, H, W) or (H, W); got {tuple(tensor.shape)}")
        if channels != 3:
            raise ValueError(f"image encoder expects 3 channels; got {channels}")
        if not self.height:
            self.height = height
        if not self.width:
            self.width = width


class ImageEncoder(MultimodalEncoder):
    """Encoder for the ``image`` modality.

    The encoder itself does not resize the input (``tensorize`` produces a
    native tensor); callers pass the tensor through ``to_features`` after
    resizing to the expected input size, or resize here by enabling
    ``resize_to``.
    """

    modality: ClassVar[Modality] = Modality.IMAGE

    def __init__(
        self,
        feature_dim: int = 256,
        resize_to: tuple[int, int] | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.resize_to = resize_to
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    @staticmethod
    def _interpolate(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Nearest-neighbour resize preserving integer channel-first layout."""
        try:
            return torch.nn.functional.interpolate(x, size=size, mode="nearest")
        except TypeError:  # pragma: no cover - very old torch
            return torch.nn.functional.interpolate(x, size=size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, feature_dim)."""
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"image encoder expects (B, 3, H, W); got {tuple(x.shape)}")
        if self.resize_to is not None:
            x = self._interpolate(x, self.resize_to)
        return self.head(self.backbone(x))

    def to_features(self, inputs: object) -> torch.Tensor:
        if isinstance(inputs, ImageDocument):
            return self.forward(tensorize(inputs.data, dtype="float32")[None])
        return self.forward(tensorize(inputs, dtype="float32"))

    def encode(self, inputs: Any) -> torch.Tensor:
        """Accepts tensors, ``ImageDocument``s, or ``(H, W)`` numpy arrays."""
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, ImageDocument):
            return self.to_features(inputs)
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty image batch")
        if len(items) == 1:
            return self.to_features(items[0])
        batch = [
            tensorize(i.data if isinstance(i, ImageDocument) else i, dtype="float32") for i in items
        ]
        return self.forward(torch.stack(batch))

    def encode_image(self, image: object) -> torch.Tensor:
        return self.encode(image)


__all__ = [
    "ImageDocument",
    "ImageEncoder",
]
