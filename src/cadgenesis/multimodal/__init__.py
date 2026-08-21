"""cadgenesis.multimodal
=======================
Pillar 3 — Multimodal Understanding.

The package provides one shared engineering embedding space for all eleven
modalities (TEXT, CAD, DRAWING, SKETCH, IMAGE, PDF, POINT_CLOUD, MESH,
AUDIO, VIDEO, SENSOR), per-modality encoders, cross-modal attention and
five fusion strategies.

Public entry point:

.. code-block:: python

    system = MultimodalSystem(embed_dim=256)
    result = system.encode({Modality.TEXT: ["bracket"], Modality.CAD: doc})
"""

from cadgenesis.multimodal.common import (
    ALL_MODALITIES,
    DEFAULT_FEATURE_DIMS,
    MODALITIES,
    Modality,
    ModalitySpec,
    modality_from_name,
    modality_specs,
    supported_families,
)
from cadgenesis.multimodal.cross_modal import (
    CrossModalAttention,
    CrossModalEngine,
    CrossModalLayer,
    CrossModalLayerRegistry,
    CrossModalResult,
)
from cadgenesis.multimodal.embeddings import (
    FeatureNormalization,
    ModalityAdapter,
    ModalityAdapterRegistry,
    ProjectionHead,
    SharedEngineeringEmbeddingSpace,
)
from cadgenesis.multimodal.fusion import FusionEngine, FusionResult, FusionStrategy
from cadgenesis.multimodal.integration import MultimodalIntegrator
from cadgenesis.multimodal.multimodal import MultimodalEncoding, MultimodalSystem

__all__ = [
    "ALL_MODALITIES",
    "DEFAULT_FEATURE_DIMS",
    "MODALITIES",
    "CrossModalAttention",
    "CrossModalEngine",
    "CrossModalLayer",
    "CrossModalLayerRegistry",
    "CrossModalResult",
    "FeatureNormalization",
    "FusionEngine",
    "FusionResult",
    "FusionStrategy",
    "Modality",
    "ModalityAdapter",
    "ModalityAdapterRegistry",
    "ModalitySpec",
    "MultimodalEncoding",
    "MultimodalIntegrator",
    "MultimodalSystem",
    "ProjectionHead",
    "SharedEngineeringEmbeddingSpace",
    "modality_from_name",
    "modality_specs",
    "supported_families",
]
