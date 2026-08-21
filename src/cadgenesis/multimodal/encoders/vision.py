"""cadgenesis.multimodal.encoders.vision
======================================
Trainable CNN vision encoder (Pillar 3).

:class:`VisionEncoderCNN` is a small convolutional stack (four pooling
stages, 3 -> 64 -> 128 -> 256 -> 512 channels) followed by global adaptive
average pooling, a dropout layer and a linear head.  It maps a batch of
images ``(B, 3, H, W)`` to raw features ``(B, feature_dim)`` and serves the
``image`` modality of the shared engineering embedding space.

``forward`` accepts any spatial size ``(H, W)`` because the global
:class:`torch.nn.AdaptiveAvgPool2d` collapses spatial dims before the head.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize

try:
    from PIL import Image as _PILImage
except ImportError:  # pragma: no cover - PIL is optional
    _PILImage = None

try:
    from torchvision import transforms as _tv_transforms
except ImportError:  # pragma: no cover - torchvision is optional
    _tv_transforms = None


class VisionEncoderCNN(MultimodalEncoder):
    """CNN encoder for the ``image`` modality."""

    modality: ClassVar[Modality] = Modality.IMAGE

    def __init__(
        self,
        feature_dim: int = 512,
        input_channels: int = 3,
        image_size: int = 224,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        #: Expected spatial size for inputs not relying on adaptive pooling.
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W) -> (B, feature_dim)``.

        Any ``(H, W)`` is accepted thanks to the global adaptive pooling;
        ``image_size`` is provided as a convention for callers that want a
        canonical size.
        """
        if x.dim() != 4:
            raise ValueError(
                f"vision encoder expects (B, C, H, W) image tensors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        """Encode a tensor batch, a list of images, or a single image.

        ``inputs`` may be a ``(B, 3, H, W)`` tensor, a list of tensors /
        numpy arrays / nested lists (each item ``(3, H, W)``), or a single
        ``(3, H, W)`` image.  PIL images are converted through torchvision
        when both PIL and torchvision are installed; otherwise a
        :class:`RuntimeError` asks the caller to pass tensors.
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            return self.forward(inputs)
        if _PILImage is not None and isinstance(inputs, _PILImage.Image):
            return self.forward(self._pil_to_tensor(inputs).unsqueeze(0))
        if (
            _PILImage is not None
            and isinstance(inputs, (list, tuple))
            and inputs
            and all(isinstance(item, _PILImage.Image) for item in inputs)
        ):
            tensors = [self._pil_to_tensor(item) for item in inputs]
            return self.forward(torch.stack(tensors))
        if isinstance(inputs, (list, tuple)) and inputs:
            first = inputs[0]
            if isinstance(first, (torch.Tensor, list, tuple)) or hasattr(first, "shape"):
                tensors = [tensorize(item, dtype="float32") for item in inputs]
                tensor = torch.stack(tensors)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                return self.forward(tensor)
        tensor = tensorize(inputs, dtype="float32")
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return self.forward(tensor)

    @staticmethod
    def _pil_to_tensor(image: Any) -> torch.Tensor:
        if _tv_transforms is None:
            raise RuntimeError(
                "PIL images require torchvision; pass a (B, 3, H, W) tensor "
                "or a list of (3, H, W) tensors instead"
            )
        return _tv_transforms.ToTensor()(image)


__all__ = ["VisionEncoderCNN"]
