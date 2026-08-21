"""cadgenesis.multimodal.encoders.mesh
=====================================
Mesh encoder (STL, OBJ, GLTF, PLY).

A mesh is normalised into a :class:`MeshDocument` holding vertices
``(V, 3)`` and triangles ``(T, 3)`` (with an ``index_into`` helper that lets
3rd-party loaders (trimesh, open3d, numpy-stl) convert their data).  The
encoder computes translation-invariant vertex statistics and a fixed
occupancy grid over the object space, then maps the descriptor through an
MLP into the shared raw feature space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize

_GRID_SIZE = 12
_GRID_DIM = _GRID_SIZE * _GRID_SIZE * _GRID_SIZE
_STAT_DIM = 14
_DESCRIPTOR_SIZE = _GRID_DIM + _STAT_DIM


@dataclass
class MeshDocument:
    """Normalised mesh: vertices (V, 3), triangles (T, 3)."""

    vertices: Any
    triangles: Any | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.vertices = tensorize(self.vertices, dtype="float32")
        if self.vertices.dim() != 2 or self.vertices.shape[-1] != 3:
            raise ValueError(f"mesh vertices must be (V, 3); got {tuple(self.vertices.shape)}")
        if self.triangles is not None:
            self.triangles = tensorize(self.triangles, dtype="long")
            if self.triangles.dim() != 2 or self.triangles.shape[-1] != 3:
                raise ValueError(
                    f"mesh triangles must be (T, 3); got {tuple(self.triangles.shape)}"
                )

    @classmethod
    def from_vertices(
        cls,
        vertices: Any,
        triangles: Any | None = None,
        name: str = "",
    ) -> MeshDocument:
        """Construct from raw vertex/triangle arrays (compatible with trimesh,
        open3d and numpy-stl data)."""
        return cls(vertices=vertices, triangles=triangles, name=name)


def parse_mesh_file(path: str | Path) -> MeshDocument:
    """Parse a mesh file on disk.

    Uses ``trimesh`` or ``numpy-stl`` when available; otherwise falls back
    to a tiny built-in STL/OBJ reader.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".stl", ".obj", ".ply", ".gltf", ".glb"):
        try:
            import trimesh  # type: ignore[import-not-found]

            mesh = trimesh.load(str(path))
            vertices = mesh.vertices.astype("float32")
            triangles = (
                mesh.faces.astype("int64") if hasattr(mesh, "faces") and mesh.faces.size else None
            )
            return MeshDocument(vertices=vertices, triangles=triangles, name=path.name)
        except ImportError:
            pass

    if ext == ".stl":
        try:
            import numpy as np
            import stl  # type: ignore[import-not-found]

            loaded = stl.mesh.Mesh.from_file(str(path))
            return MeshDocument(
                vertices=np.asarray(loaded.vectors.reshape(-1, 3), dtype="float32"),
                triangles=np.arange(len(loaded.vectors) * 3).reshape(-1, 3),
                name=path.name,
            )
        except ImportError:
            pass
        stl_vertices: list[list[float]] = []
        stl_triangles: list[list[int]] = []
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("vertex "):
                stl_vertices.append([float(v) for v in stripped.split()[1:4]])
            elif stripped.startswith("facet "):
                start = len(stl_vertices)
                stl_triangles.append([start, start + 1, start + 2])
        return MeshDocument(
            vertices=stl_vertices or [[0.0, 0.0, 0.0]],
            triangles=stl_triangles or None,
            name=path.name,
        )

    if ext == ".obj":
        obj_vertices: list[list[float]] = []
        obj_triangles: list[list[int]] = []
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("v "):
                obj_vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                if len(faces) >= 3:
                    obj_triangles.extend(
                        [faces[0], faces[i], faces[i + 1]] for i in range(1, len(faces) - 1)
                    )
        return MeshDocument(
            vertices=obj_vertices or [[0.0, 0.0, 0.0]],
            triangles=obj_triangles or None,
            name=path.name,
        )

    raise ValueError(f"unsupported mesh format {ext!r}")


def _voxelize(vertices: torch.Tensor, grid: int) -> torch.Tensor:
    if vertices.numel() == 0:
        return torch.zeros(grid * grid * grid, dtype=torch.float32)
    center = vertices.mean(dim=0)
    span = (vertices - center).abs().max().clamp_min(1e-6)
    scaled = (vertices - center) / span
    indices = ((scaled * 0.5 + 0.5) * (grid - 1)).long().clamp(0, grid - 1)
    linear = indices[:, 0] * grid * grid + indices[:, 1] * grid + indices[:, 2]
    occupancy = torch.zeros(grid * grid * grid, dtype=torch.float32)
    occupancy.scatter_add_(0, linear, torch.ones_like(linear, dtype=torch.float32))
    return (occupancy > 0).float()


def mesh_document_descriptor(document: MeshDocument) -> torch.Tensor:
    """Deterministic descriptor: occupancy grid + vertex statistics."""
    vertices = document.vertices
    grid = _voxelize(vertices, _GRID_SIZE)

    stats = torch.zeros(_STAT_DIM, dtype=torch.float32)
    if vertices.numel() > 0:
        centered = vertices - vertices.mean(dim=0)
        covariance = centered.t() @ centered / max(vertices.shape[0], 1)
        covariance = covariance + torch.eye(3) * 1e-8
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        stats[0] = math.log1p(vertices.shape[0])
        stats[1] = vertices.abs().max().item()
        stats[2] = (eigenvalues[2] / max(eigenvalues.sum(), 1e-8)).item()
        stats[3] = ((eigenvalues[2] - eigenvalues[1]) / max(eigenvalues[2], 1e-8)).item()
        stats[4] = ((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[1], 1e-8)).item()
        stats[5] = centered.norm(dim=1).mean().item()
        stats[6] = centered.norm(dim=1).std().item()
        stats[7] = (grid > 0).float().sum().item() / _GRID_DIM
        stats[8] = 1.0 if document.triangles is not None else 0.0
        stats[9] = (
            math.log1p(document.triangles.shape[0]) if document.triangles is not None else 0.0
        )
        stats[10] = math.log1p(centered.norm(dim=1).max().item())
        stats[11] = vertices[:, 2].std().item() if vertices.shape[0] > 1 else 0.0
        stats[12] = vertices.shape[0] / max(grid.float().sum().item(), 1e-6)
        stats[13] = 1.0 if document.name else 0.0

    return torch.cat([grid, stats])


class MeshEncoder(MultimodalEncoder):
    """Encoder for the ``mesh`` modality."""

    modality: ClassVar[Modality] = Modality.MESH

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
                f"mesh encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, (str, Path)):
            inputs = [parse_mesh_file(inputs)]
        elif isinstance(inputs, MeshDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty mesh batch")
        descriptors = torch.stack([mesh_document_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = [
    "MeshDocument",
    "MeshEncoder",
    "mesh_document_descriptor",
    "parse_mesh_file",
]
