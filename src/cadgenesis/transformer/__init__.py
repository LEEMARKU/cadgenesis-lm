"""
cadgenesis.transformer
===============
Geometry-Aware Transformer & Self-Designing Transformer models for CADGenesis-LM v2.0.

Pillar 1 (v6.0) additions:
- sparse attention (local/global/sliding-window/block-sparse)
- multi-scale attention (local + medium + global)
- specialized MoE (geometry/manufacturing/reasoning/simulation/optimization)
- hierarchical transformer (Planner→Geometry→Constraint→Execution→Validation)
- dynamic computation routing (early exit + computation budgeting)
- configurable transformer evolution framework (versioning/plugins/registries)
"""

from cadgenesis.transformer.attention import (
    AgentAttention,
    ConstraintAttention,
    GeometryAttention,
    MemoryAttention,
    MultiHeadAttentionMixture,
    SelfAttention,
    UncertaintyAttention,
)
from cadgenesis.transformer.dynamic_routing import (
    ComputationBudget,
    DynamicRoutingController,
    EarlyExitGate,
)
from cadgenesis.transformer.efficient_attention import (
    LinearAttention,
    SDPASelfAttention,
    build_self_attention,
)
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
from cadgenesis.transformer.hierarchical_transformer import (
    STAGE_NAMES,
    HierarchicalCADTransformer,
)
from cadgenesis.transformer.interaction import FeatureInteractionLayer
from cadgenesis.transformer.moe import SparseMoEFFN
from cadgenesis.transformer.multi_scale_attention import MultiScaleAttention
from cadgenesis.transformer.positional import (
    ALiBiBias,
    GeometryPositionalEncoding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)
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
from cadgenesis.transformer.transformer_block import CADTransformerBlock, RMSNorm, SwiGLU

__all__ = [
    "DEFAULT_DOMAIN_EXPERTS",
    "SPARSE_PATTERNS",
    "STAGE_NAMES",
    "ALiBiBias",
    "AgentAttention",
    "ArchitectureVersion",
    "CADTransformerBlock",
    "ComputationBudget",
    "ConfigurationDrivenBuilder",
    "ConstraintAttention",
    "DomainExpert",
    "DynamicRoutingController",
    "EarlyExitGate",
    "ExperimentRecord",
    "ExperimentRegistry",
    "FeatureInteractionLayer",
    "GeometryAttention",
    "GeometryAwareTransformer",
    "GeometryPositionalEncoding",
    "HierarchicalCADTransformer",
    "LayerRegistry",
    "LinearAttention",
    "MemoryAttention",
    "MultiHeadAttentionMixture",
    "MultiScaleAttention",
    "Plugin",
    "PluginManager",
    "RMSNorm",
    "RegistryStack",
    "RotaryEmbedding",
    "SDPASelfAttention",
    "SelfAttention",
    "SelfDesigningTransformer",
    "SinusoidalPositionalEncoding",
    "SparseAttentionPattern",
    "SparseMoEFFN",
    "SparseSelfAttention",
    "SpecializedMoEFFN",
    "SwiGLU",
    "UncertaintyAttention",
    "VersionedArchitecture",
    "build_self_attention",
    "build_sparse_attention",
    "global_registry",
    "hash_architecture",
    "register_expert_type",
    "register_layer",
    "registered_expert_types",
    "sparse_attention_mask",
]
