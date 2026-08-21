"""cadgenesis.transformer.transformer
==================================
Facade module aggregating the public transformer API for CADGenesis-LM v6.0.

This module intentionally re-exports the canonical implementations so that
``from cadgenesis.transformer.transformer import GeometryAwareTransformer``
remains stable.
"""

from cadgenesis.transformer.decoder import DecoderStack
from cadgenesis.transformer.dynamic_routing import (
    ComputationBudget,
    DynamicRoutingController,
    EarlyExitGate,
)
from cadgenesis.transformer.embeddings import (
    CombinedInputEmbedding,
    TokenEmbedding,
    TypeEmbedding,
)
from cadgenesis.transformer.encoder import EncoderStack
from cadgenesis.transformer.evolution import (
    ArchitectureVersion,
    ConfigurationDrivenBuilder,
    ExperimentRecord,
    ExperimentRegistry,
    LayerRegistry,
    Plugin,
    PluginManager,
    RegistryStack,
    VersionedArchitecture,
    global_registry,
    hash_architecture,
    register_layer,
)
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.heads import (
    ConfidenceHead,
    LMHead,
    OutputHeads,
)
from cadgenesis.transformer.hierarchical_transformer import (
    STAGE_NAMES,
    HierarchicalCADTransformer,
)
from cadgenesis.transformer.losses import (
    CADSequenceLoss,
    ConfidenceLoss,
    MaskedCrossEntropyLoss,
)
from cadgenesis.transformer.multi_scale_attention import MultiScaleAttention
from cadgenesis.transformer.self_designing import SelfDesigningTransformer
from cadgenesis.transformer.sparse_attention import (
    SPARSE_PATTERNS,
    SparseAttentionPattern,
    SparseSelfAttention,
    build_sparse_attention,
    sparse_attention_mask,
)
from cadgenesis.transformer.specialized_moe import (
    DEFAULT_DOMAIN_EXPERTS,
    DomainExpert,
    SpecializedMoEFFN,
    register_expert_type,
    registered_expert_types,
)
from cadgenesis.transformer.transformer_block import (
    CADTransformerBlock,
    RMSNorm,
    SwiGLU,
)

__all__ = [
    "DEFAULT_DOMAIN_EXPERTS",
    "SPARSE_PATTERNS",
    "STAGE_NAMES",
    "ArchitectureVersion",
    "CADSequenceLoss",
    "CADTransformerBlock",
    "CombinedInputEmbedding",
    "ComputationBudget",
    "ConfidenceHead",
    "ConfidenceLoss",
    "ConfigurationDrivenBuilder",
    "DecoderStack",
    "DomainExpert",
    "DynamicRoutingController",
    "EarlyExitGate",
    "EncoderStack",
    "ExperimentRecord",
    "ExperimentRegistry",
    "GeometryAwareTransformer",
    "HierarchicalCADTransformer",
    "LMHead",
    "LayerRegistry",
    "MaskedCrossEntropyLoss",
    "MultiScaleAttention",
    "OutputHeads",
    "Plugin",
    "PluginManager",
    "RMSNorm",
    "RegistryStack",
    "SelfDesigningTransformer",
    "SparseAttentionPattern",
    "SparseSelfAttention",
    "SpecializedMoEFFN",
    "SwiGLU",
    "TokenEmbedding",
    "TypeEmbedding",
    "VersionedArchitecture",
    "build_sparse_attention",
    "global_registry",
    "hash_architecture",
    "register_expert_type",
    "register_layer",
    "registered_expert_types",
    "sparse_attention_mask",
]
