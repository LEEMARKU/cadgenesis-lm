"""cadgenesis.multimodal.encoders.video
=====================================
Video encoder (design review walkthroughs, assembly animation).

A video is normalised into a :class:`VideoDocument` holding a list of frames
(each a ``(3, H, W)`` tensor, normalised at construction time) plus the
frame rate.  The encoder computes a fixed-length sequence of per-frame
temporal / appearance statistics (mean intensity, optical-flow magnitude
proxy, frame-to-frame difference) and maps the fixed-size descriptor through
an MLP into the shared raw feature space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize

_MAX_FRAMES = 32
_FRAME_STAT_DIM = 4
_STAT_DIM = 8
_DESCRIPTOR_SIZE = _MAX_FRAMES * _FRAME_STAT_DIM + _STAT_DIM


@dataclass
class VideoDocument:
    """Normalised video: list of (3, H, W) frame tensors + fps."""

    frames: list[Any]
    fps: float = 30.0
    source: str = "review"

    def __post_init__(self) -> None:
        normalised: list[torch.Tensor] = []
        for frame in self.frames:
            tensor = tensorize(frame, dtype="float32")
            if tensor.dim() == 2:
                tensor = tensor[None]
            if tensor.dim() != 3 or tensor.shape[0] != 3:
                raise ValueError(f"video frames must be (3, H, W); got {tuple(tensor.shape)}")
            normalised.append(tensor)
        self.frames = normalised

    @classmethod
    def from_path(cls, path: str) -> VideoDocument:
        """Load a video file with ``decord`` or ``av`` when available.

        Raises ``ImportError`` if neither backend is installed — the caller
        is expected to handle the missing-backend case explicitly.
        """
        frames: list[torch.Tensor] = []
        try:
            import decord  # type: ignore[import-not-found]

            reader = decord.VideoReader(path)
            for index in range(0, len(reader), max(len(reader) // _MAX_FRAMES, 1)):
                frame = reader[index].asnumpy()
                frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
            fps = float(reader.get_avg_fps())
            return cls(frames=frames, fps=fps)
        except ImportError:
            pass
        try:
            import av  # type: ignore[import-not-found]

            container = av.open(path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            for frame in container.decode(stream):
                if len(frames) >= _MAX_FRAMES:
                    break
                image = frame.to_image()
                frames.append(
                    torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float()
                    / 255.0
                )
            return cls(frames=frames, fps=fps)
        except ImportError:
            pass
        raise ImportError(
            "video decoding requires 'decord' or 'av'; install one of them "
            "or build VideoDocument(frames=[...]) manually"
        )


def _video_descriptor(document: VideoDocument) -> torch.Tensor:
    """Deterministic descriptor of a video.

    Per frame (over the fixed max-frame grid): mean, std, frame-difference
    (motion proxy), max-intensity.  Plus 8 global statistics (frame count,
    fps, motion energy, mean spatial size, ...).
    """
    descriptor = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    if not document.frames:
        return descriptor

    frames = document.frames[:_MAX_FRAMES]
    grid = torch.linspace(0, len(frames) - 1, _MAX_FRAMES).long()
    selected = [frames[int(i)] for i in grid]

    for i, frame in enumerate(selected):
        base = i * _FRAME_STAT_DIM
        descriptor[base + 0] = frame.mean().item()
        descriptor[base + 1] = frame.std().item()
        descriptor[base + 2] = frame.max().item()
        if i > 0:
            descriptor[base + 3] = (frame - selected[i - 1]).abs().mean().item()
        else:
            descriptor[base + 3] = 0.0

    offset = _MAX_FRAMES * _FRAME_STAT_DIM
    descriptor[offset + 0] = math.log1p(len(document.frames))
    descriptor[offset + 1] = math.log1p(document.fps)
    motion = sum((frames[i] - frames[i - 1]).abs().mean().item() for i in range(1, len(frames)))
    descriptor[offset + 2] = motion / max(len(frames) - 1, 1)
    descriptor[offset + 3] = 1.0 if document.source == "review" else 0.0
    descriptor[offset + 4] = frames[0].shape[1]
    descriptor[offset + 5] = frames[0].shape[2]
    descriptor[offset + 6] = sum(f.mean().item() for f in frames) / len(frames)
    descriptor[offset + 7] = max(
        len(frames) / _MAX_FRAMES,
        len(frames) / max(len(document.frames), 1),
    )
    return descriptor


class VideoEncoder(MultimodalEncoder):
    """Encoder for the ``video`` modality."""

    modality: ClassVar[Modality] = Modality.VIDEO

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
                f"video encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, VideoDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty video batch")
        descriptors = torch.stack([_video_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = ["VideoDocument", "VideoEncoder"]
