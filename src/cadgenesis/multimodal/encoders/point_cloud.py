"""cadgenesis.multimodal.encoders.point_cloud
============================================
Point-cloud encoder (LiDAR, RGB-D, tactile point clouds).

A point cloud is normalised into a :class:`PointCloudDocument` holding a
``(N, 3)`` float array plus the source type.  The encoder computes
translation-invariant shape statistics (bounds, density, variance, moments,
eigenvalue-based shape descriptors), discretises the point set into a fixed
occupancy grid, and maps the resulting fixed-size descriptor through an MLP
into the shared raw feature space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize

_GRID_SIZE = 16
_GRID_DIM = _GRID_SIZE * _GRID_SIZE * _GRID_SIZE
_STAT_DIM = 16
_DESCRIPTOR_SIZE = _GRID_DIM + _STAT_DIM


@dataclass
class PointCloudDocument:
    """Normalised point cloud (N, 3) with source metadata."""

    points: Any
    source: str = "lidar"
    normals: Any | None = None

    def __post_init__(self) -> None:
        points = tensorize(self.points, dtype="float32")
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(f"point cloud must be (N, 3); got {tuple(points.shape)}")
        self.points = points


def _voxelize(points: torch.Tensor, grid: int) -> torch.Tensor:
    """Occupancy grid of a point cloud (translation + scale invariant)."""
    if points.numel() == 0:
        return torch.zeros(grid * grid * grid, dtype=torch.float32)
    center = points.mean(dim=0)
    centered = points - center
    span = centered.abs().max().clamp_min(1e-6)
    scaled = centered / span
    indices = ((scaled * 0.5 + 0.5) * (grid - 1)).long().clamp(0, grid - 1)
    linear = indices[:, 0] * grid * grid + indices[:, 1] * grid + indices[:, 2]
    occupancy = torch.zeros(grid * grid * grid, dtype=torch.float32)
    occupancy.scatter_add_(0, linear, torch.ones_like(linear, dtype=torch.float32))
    occupancy = (occupancy > 0).float()
    return occupancy


def point_cloud_descriptor(document: PointCloudDocument) -> torch.Tensor:
    """Deterministic descriptor: occupancy grid + shape statistics."""
    points = document.points
    grid = _voxelize(points, _GRID_SIZE)

    stats = torch.zeros(_STAT_DIM, dtype=torch.float32)
    if points.numel() > 0:
        centered = points - points.mean(dim=0)
        covariance = centered.t() @ centered / max(points.shape[0], 1)
        covariance = covariance + torch.eye(3) * 1e-8
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        stats[0] = math.log1p(points.shape[0])
        stats[1] = points.abs().max().item()
        stats[2] = (eigenvalues[2] / max(eigenvalues.sum(), 1e-8)).item()
        stats[3] = ((eigenvalues[2] - eigenvalues[1]) / max(eigenvalues[2], 1e-8)).item()
        stats[4] = ((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[1], 1e-8)).item()
        stats[5] = centered.norm(dim=1).mean().item()
        stats[6] = centered.norm(dim=1).std().item()
        stats[7] = (grid > 0).float().sum().item() / _GRID_DIM
        stats[8] = points[:, 2].std().item() if points.shape[0] > 1 else 0.0
        stats[9] = points.shape[0] / max(grid.float().sum().item(), 1e-6)
        stats[10] = 1.0 if document.source == "lidar" else 0.0
        stats[11] = 1.0 if document.source == "rgb_d" else 0.0
        stats[12] = 1.0 if document.source == "tactile" else 0.0
        stats[13] = 1.0 if document.normals is not None else 0.0
        stats[14] = math.log1p(centered.norm(dim=1).max().item())
        stats[15] = 1.0 if points.shape[0] >= 1000 else float(points.shape[0]) / 1000.0

    return torch.cat([grid, stats])


class PointCloudEncoder(MultimodalEncoder):
    """Encoder for the ``point_cloud`` modality."""

    modality: ClassVar[Modality] = Modality.POINT_CLOUD

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.net = nn.Sequential(
            nn.Linear(_DESCRIPTOR_SIZE, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != _DESCRIPTOR_SIZE:
            raise ValueError(
                f"point-cloud encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; "
                f"got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, PointCloudDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty point-cloud batch")
        descriptors = torch.stack([point_cloud_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = [
    "PointCloudDocument",
    "PointCloudEncoder",
    "point_cloud_descriptor",
]
