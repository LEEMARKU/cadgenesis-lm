"""cadgenesis.multimodal.encoders
================================
Encoder registry for all eleven CADGenesis modalities.

Each encoder is a :class:`MultimodalEncoder` subclass mapping raw inputs
(structured documents, files, tensors) into the *raw* feature space; the
:class:`SharedEngineeringEmbeddingSpace` projects those features into the
single shared engineering embedding space.

The :func:`encoder_registry` returns a ``{Modality: encoder-factory}`` map,
and :func:`build_encoders` instantiates one encoder per modality using the
feature dims supplied by :func:`cadgenesis.multimodal.common.modality_specs`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cadgenesis.multimodal.common import Modality, modality_specs
from cadgenesis.multimodal.encoders.audio import AudioEncoder
from cadgenesis.multimodal.encoders.base import MultimodalEncoder
from cadgenesis.multimodal.encoders.cad import CADEncoder
from cadgenesis.multimodal.encoders.drawing import DrawingEncoder
from cadgenesis.multimodal.encoders.image import ImageEncoder
from cadgenesis.multimodal.encoders.mesh import MeshEncoder
from cadgenesis.multimodal.encoders.pdf import PDFEncoder
from cadgenesis.multimodal.encoders.pointcloud import PointCloudEncoder
from cadgenesis.multimodal.encoders.sensor import SensorEncoder
from cadgenesis.multimodal.encoders.sketch import SketchEncoder
from cadgenesis.multimodal.encoders.text import TextEncoder
from cadgenesis.multimodal.encoders.video import VideoEncoder
from cadgenesis.multimodal.encoders.vision import VisionEncoderCNN

EncoderFactory = Callable[..., MultimodalEncoder]

_FACTORIES: dict[Modality, EncoderFactory] = {
    Modality.TEXT: TextEncoder,
    Modality.CAD: CADEncoder,
    Modality.DRAWING: DrawingEncoder,
    Modality.SKETCH: SketchEncoder,
    Modality.IMAGE: VisionEncoderCNN,
    Modality.PDF: PDFEncoder,
    Modality.POINT_CLOUD: PointCloudEncoder,
    Modality.MESH: MeshEncoder,
    Modality.AUDIO: AudioEncoder,
    Modality.VIDEO: VideoEncoder,
    Modality.SENSOR: SensorEncoder,
}

__all__ = [
    "AudioEncoder",
    "CADEncoder",
    "DrawingEncoder",
    "EncoderFactory",
    "ImageEncoder",
    "MeshEncoder",
    "MultimodalEncoder",
    "PDFEncoder",
    "PointCloudEncoder",
    "SensorEncoder",
    "SketchEncoder",
    "TextEncoder",
    "VideoEncoder",
    "VisionEncoderCNN",
    "build_encoders",
    "encoder_registry",
]


def encoder_registry() -> dict[Modality, EncoderFactory]:
    """Return the modality -> encoder-factory registry."""
    return dict(_FACTORIES)


def build_encoders(
    feature_dims: dict[Modality, int] | None = None,
    **kwargs: Any,
) -> dict[Modality, MultimodalEncoder]:
    """Instantiate one encoder per modality.

    ``feature_dims`` maps each modality to its raw feature dimension;
    defaults come from :func:`modality_specs`.  Any additional ``kwargs``
    (e.g. ``dropout=0.05``) are forwarded to every encoder constructor.
    """
    specs = modality_specs(feature_dims)
    dims = {spec.modality: spec.feature_dim for spec in specs}
    encoders: dict[Modality, MultimodalEncoder] = {}
    for modality, factory in _FACTORIES.items():
        dim = dims[modality]
        encoders[modality] = factory(feature_dim=dim, **kwargs)
    return encoders
