"""cadgenesis.multimodal.encoders.pointcloud
===========================================
PointNet-style encoder for the ``point_cloud`` modality (Pillar 3).

:class:`PointCloudEncoder` applies a shared MLP to every point
(``Linear point_dim -> 64 -> 128 -> 256`` with ReLU and ``BatchNorm1d``
over the flattened ``(B * N, point_dim)`` tensor), pools the per-point
features with a max operation over the point axis, and maps the pooled
vector through a ``256 -> 512 -> feature_dim`` head.  It maps a batch of
point clouds ``(B, N, point_dim)`` to raw features ``(B, feature_dim)``
and is permutation-invariant in the point order.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize


class PointCloudEncoder(MultimodalEncoder):
    """PointNet-style encoder for the ``point_cloud`` modality."""

    modality: ClassVar[Modality] = Modality.POINT_CLOUD

    def __init__(
        self,
        feature_dim: int = 512,
        point_dim: int = 3,
        num_points: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        #: Dimensionality of each point (x, y, z for 3D scans).
        self.point_dim = point_dim
        #: Optional expected point count per cloud; ``None`` accepts any ``N``.
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, point_dim) -> (B, feature_dim)``."""
        if x.dim() != 3:
            raise ValueError(
                f"point cloud encoder expects (B, N, {self.point_dim}) tensors; "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.point_dim:
            raise ValueError(
                f"point cloud encoder expects point_dim={self.point_dim}; got {tuple(x.shape)}"
            )
        batch_size, num_points, _ = x.shape
        # BatchNorm1d requires > 1 sample per channel while training; a single
        # request (B == 1) falls back to running stats (inference semantics).
        was_training = self.training
        if self.training and batch_size == 1:
            self.eval()
        try:
            features = self.mlp(x.reshape(-1, self.point_dim))
            features = features.reshape(batch_size, num_points, -1)
            pooled = features.max(dim=1).values
            return self.head(pooled)
        finally:
            self.train(was_training)

    def encode(self, inputs: Any) -> torch.Tensor:
        """Encode a tensor batch, a list of clouds, or a single cloud.

        ``inputs`` may be a ``(B, N, point_dim)`` tensor, a list of tensors /
        numpy arrays / nested lists (each item ``(N, point_dim)``), or a
        single ``(N, point_dim)`` cloud.
        """
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, (list, tuple)) and inputs:
            first = inputs[0]
            if isinstance(first, (torch.Tensor, list, tuple)) or hasattr(first, "shape"):
                tensors = [tensorize(item, dtype="float32") for item in inputs]
                return self.forward(torch.stack(tensors))
        return self.forward(tensorize(inputs, dtype="float32"))


__all__ = ["PointCloudEncoder"]
