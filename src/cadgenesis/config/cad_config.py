"""
CADConfig — Single source of truth for all CADGenesis-LM v2.0 hyperparameters.

Design principles:
  - Every configurable value lives here; nothing is hard-coded in sub-modules.
  - Dataclass with post_init validation so misconfiguration fails fast.
  - Serializable to/from JSON and YAML for experiment tracking.
  - Nested config groups mirror the phase architecture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sub-configurations (one per major subsystem)
# ---------------------------------------------------------------------------


@dataclass
class TokenizerConfig:
    """Autonomous CAD Tokenizer configuration."""

    # ---- Text side ----
    lang_vocab_size: int = 32_000  # BPE vocabulary target size
    lang_max_len: int = 512  # Max language token sequence length

    # ---- Geometry / numeric quantization ----
    num_bins: int = 256  # Quantization resolution for continuous params
    param_min: float = 0.0  # Minimum parametric value (mm)
    param_max: float = 1_000.0  # Maximum parametric value (mm)
    angle_bins: int = 360  # Degree quantization (1° resolution)

    # ---- CAD feature vocabulary ----
    max_feature_tokens: int = 512  # Max tokens per CAD feature definition
    max_cad_seq_len: int = 1_024  # Max tokens in a full CAD sequence

    # ---- Token type families (must sum to total vocab minus specials) ----
    # Each family reserves a contiguous ID range in the vocabulary.
    geometry_token_slots: int = 512  # Geometric primitive + B-Rep tokens
    feature_token_slots: int = 512  # CAD feature operation tokens
    constraint_token_slots: int = 256  # Parametric constraint tokens
    material_token_slots: int = 256  # Material + property tokens
    assembly_token_slots: int = 256  # Assembly relationship tokens
    manufacturing_token_slots: int = 256  # Manufacturing process tokens
    simulation_token_slots: int = 256  # Simulation / physics tokens
    numeric_token_slots: int = 1024  # Quantized numeric parameter tokens
    special_token_slots: int = 64  # <pad>, <bos>, <eos>, <sep>, <mask>, etc.


@dataclass
class ModelConfig:
    """
    Geometry-Aware Transformer configuration (Phase 2 target).

    The defaults describe a **lean, production-oriented encoder-decoder**:
    only standard self-attention + encoder-decoder (geometry) cross-attention,
    with all experimental subsystems (multi-agent system, layer-integrated
    memory pools, neuro-symbolic rules, RLAIF reward model) disabled.  Enable
    them explicitly when the added complexity is justified by evidence; the
    specialized attention heads are forced to 0 unless their governing
    subsystem flag is on.
    """

    d_model: int = 1_024
    nhead: int = 16
    num_encoder_layers: int = 12
    num_decoder_layers: int = 12
    dim_feedforward: int = 4_096
    dropout: float = 0.1
    max_seq_len: int = 2_048
    rope_theta: float = 10_000.0  # RoPE base frequency

    # Long-context RoPE scaling (YaRN / NTK / linear).  "none" = legacy.
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    max_position_embeddings: int = 2_048

    # Attention head type counts (must sum to nhead).  The exotic heads
    # (constraint / memory / agent / uncertainty) default to 0 for the lean
    # architecture; agent & memory heads are additionally zeroed at model
    # build time unless their subsystem flag below is enabled.
    self_attn_heads: int = 8
    geometry_attn_heads: int = 8
    constraint_attn_heads: int = 0
    memory_attn_heads: int = 0
    agent_attn_heads: int = 0
    uncertainty_attn_heads: int = 0

    # Experimental subsystem switches (lean default — OFF).
    use_multi_agent_system: bool = False
    use_memory_system: bool = False
    use_neuro_symbolic_reasoning: bool = False
    use_rlaf_reward_model: bool = False
    use_confidence_head: bool = True

    # Hybrid state-space layer (Gated DeltaNet), interleaved with attention
    # blocks every ``ssm_every_n_blocks`` blocks (2025-2026 frontier).
    use_ssm: bool = False
    ssm_every_n_blocks: int = 3
    ssm_heads: int = 4

    # BitNet b1.58: ternary weights + int8 activations (CPU-friendly).
    use_bitnet: bool = False

    # Sparse Mixture-of-Experts FFN (Self-Designing extension)
    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    expert_dim: int | None = None
    expert_router_jitter: float = 0.02
    # DeepSeek-V3 style shared expert: always-active expert fused into every
    # token alongside the top-k routed experts.
    num_shared_experts: int = 0
    shared_expert_dim: int | None = None
    # DeepSeek-V3 style MoE options (aux-loss-free balancing, z-loss,
    # expert-bias, capacity-based token dropping).
    moe_aux_free_balancing: bool = False
    moe_balance_speed: float = 0.001
    moe_z_loss_weight: float = 1e-3
    moe_capacity_factor: float | None = None
    moe_drop_tokens: bool = False

    # Efficient attention optimizations (Geometry Transformer upgrade)
    # One of "math" (default/legacy), "sdpa", "flash", "linear", "gqa", "mla".
    attention_backend: str = "math"
    # GQA / MLA options (modern attention heads).
    num_kv_heads: int | None = None  # GQA: KV heads per block (None → 1)
    kv_lora_rank: int = 64  # MLA: latent KV compression rank
    qk_rope_head_dim: int = 64  # MLA: RoPE dim of queries/keys

    # Multi-token prediction (DeepSeek-V3 style) auxiliary heads.
    mtp_depth: int = 0  # 0 → disabled
    mtp_weight: float = 0.1  # weight of the MTP loss

    # Geometry positional encoding: adds learned X/Y/Z coordinate encodings
    # to token embeddings.  Requires ``geometry_coords`` at encode/decode time.
    geometry_pos_encoding: bool = False

    # Gated cross-feature interaction sub-layer inside each transformer block.
    feature_interaction: bool = False
    interaction_heads: int = 2

    # Sparse attention (Pillar 1). Off by default → identical to the legacy
    # quadratic attention. Pattern is one of "local", "global", "sliding_window",
    # "block_sparse" or "mixed" (used when ``sparse_attention_heads > 0``).
    sparse_attention: bool = False
    sparse_attention_pattern: str = "sliding_window"
    sparse_attention_heads: int = 0  # 0 → reuse self_attn_heads
    sliding_window_size: int = 128
    local_attention_size: int = 128
    num_global_tokens: int = 32
    block_size: int = 64

    # Multi-scale attention (local + medium + global heads in parallel).
    use_multi_scale_attention: bool = False
    multi_scale_heads: int = 0  # 0 → reuse self_attn_heads
    multi_scale_local_window: int = 64
    multi_scale_medium_window: int = 256

    # Hierarchical transformer: per-stage depth (Planner → Geometry →
    # Constraint → Execution → Validation).
    use_hierarchical_transformer: bool = False
    planner_layers: int = 1
    geometry_layers: int = 1
    constraint_layers: int = 1
    execution_layers: int = 1
    validation_layers: int = 1

    # Specialized mixture-of-experts with domain experts
    # ("geometry", "manufacturing", "reasoning", "simulation", "optimization").
    use_specialized_moe: bool = False
    experts_per_domain: int = 2
    top_k_domain_experts: int = 2

    # Dynamic computation routing: early exit + computation budgeting.
    early_exit_threshold: float = 0.0  # 0 disables early exit
    computation_budget: float = 1.0  # fraction of layers to run in [0, 1]

    # Configurable transformer evolution framework.
    architecture_version: str = "1.0.0"
    evolution_plugins_enabled: bool = True
    evolution_plugins: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    batch_size: int = 64
    grad_accum_steps: int = 4
    max_epochs: int = 100
    warmup_steps: int = 2_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 200
    # --- Modern training options (P0 modernization) ---
    schedule: str = "cosine"  # "cosine" | "wsd" | "linear_decay" | ...
    wsd_stable_ratio: float = 0.75  # fraction of steps at peak LR (WSD)
    wsd_decay_ratio: float = 0.25  # fraction of steps in cosine decay
    wsd_min_lr_ratio: float = 0.1  # floor LR as fraction of peak (WSD)
    use_packing: bool = False  # sequence packing collate
    packed_max_src_len: int = 256
    packed_max_tgt_len: int = 128
    label_smoothing: float = 0.0
    moe_aux_scale: float = 0.01  # load-balancing loss weight
    confidence_loss_weight: float = 0.1
    rlaf_reward_weight: float = 0.0  # RLAIF reward-maximisation weight (0 = off)
    use_fsdp: bool = False
    use_ddp: bool = False


@dataclass
class RuntimeConfig:
    """HardwareAwareRuntime configuration (v6.2)."""

    preset: str = "auto"  # "auto" | "gtx1650_4gb" | "rtx3050_8gb" | "cpu"
    enforce_preset: bool = False  # when True, planner clamps batch/seq at init


@dataclass
class LoRAConfig:
    """LoRA / QLoRA PEFT configuration (Phase 7)."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    use_qlora: bool = False
    qlora_bits: int = 4  # 4 or 8


@dataclass
class MemoryConfig:
    """Layer-Integrated Memory Pool configuration (Phase 3)."""

    embedding_dim: int = 1_024
    working_memory_slots: int = 64
    session_memory_slots: int = 512
    user_memory_slots: int = 2_048
    project_memory_slots: int = 4_096
    cad_memory_slots: int = 8_192
    engineering_memory_slots: int = 4_096
    manufacturing_memory_slots: int = 2_048
    simulation_memory_slots: int = 2_048
    retrieval_top_k: int = 16


@dataclass
class ObservabilityConfig:
    """Telemetry / monitoring / logging configuration (v6.0)."""

    log_level: str = "INFO"
    log_json: bool = False
    log_file_enabled: bool = False
    log_file_path: str = "outputs/logs/cadgenesis.log"
    telemetry_enabled: bool = True
    metrics_prefix: str = "cadgenesis"
    tracing_enabled: bool = True
    health_check_interval_s: float = 30.0
    drift_threshold: float = 0.2
    drift_bins: int = 10


@dataclass
class MultimodalConfig:
    """Multimodal Understanding (Pillar 3) configuration.

    Controls the shared engineering embedding space, the per-modality
    encoders, cross-modal attention and the fusion engine.
    """

    embed_dim: int = 256  # shared engineering embedding dimension
    projection_hidden: int = 512  # projection-head hidden width
    use_modality_adapters: bool = True  # per-modality adapter after projection
    normalize: str = "l2"  # "none" | "l2" | "layer_norm"
    dropout: float = 0.1

    # Raw feature dimensions produced by each modality encoder before the
    # shared projection head (all 11 modalities are always registered).
    text_feature_dim: int = 512
    cad_feature_dim: int = 384
    drawing_feature_dim: int = 256
    sketch_feature_dim: int = 256
    image_feature_dim: int = 256
    pdf_feature_dim: int = 384
    point_cloud_feature_dim: int = 256
    mesh_feature_dim: int = 256
    audio_feature_dim: int = 256
    video_feature_dim: int = 256
    sensor_feature_dim: int = 256

    # Cross-modal attention.
    cross_modal_heads: int = 4
    cross_modal_layers: int = 2

    # Fusion strategy: "early" | "late" | "hierarchical" | "adaptive" |
    # "attention" (see ``FusionEngine``).
    fusion_strategy: str = "attention"

    def feature_dims(self) -> dict[str, int]:
        """Map config feature-dimension fields to canonical modality names."""
        return {
            "text": self.text_feature_dim,
            "cad": self.cad_feature_dim,
            "drawing": self.drawing_feature_dim,
            "sketch": self.sketch_feature_dim,
            "image": self.image_feature_dim,
            "pdf": self.pdf_feature_dim,
            "point_cloud": self.point_cloud_feature_dim,
            "mesh": self.mesh_feature_dim,
            "audio": self.audio_feature_dim,
            "video": self.video_feature_dim,
            "sensor": self.sensor_feature_dim,
        }


@dataclass
class WorldModelConfig:
    """World Model (Pillar 4) configuration.

    Tunes the internal object representation, spatial/mechanical/functional/
    assembly reasoners, affordance and design-intent models, the world
    simulator and the hierarchical planner.
    """

    enabled: bool = True
    # Geometry tolerance used by the spatial reasoner (mm).
    spatial_tolerance: float = 1e-4
    # Default safety factor used by mechanical stability checks.
    safety_factor: float = 2.5
    # Whether the world simulator verifies manufacturability (DFM) when
    # evolving geometry.
    check_manufacturability: bool = True
    # Whether the world simulator runs full design-consistency validation.
    check_consistency: bool = True
    # Hierarchical planner: include a validation stage in generated plans.
    include_validation_stage: bool = True
    # World model writes plan summaries into the semantic memory facade when
    # integrated with one.
    memory_enabled: bool = True
    memory_pool: str = "engineering"


@dataclass
class AgentsConfig:
    """Multi-Agent Intelligence (Pillar 5) configuration.

    Controls the agent platform: registry, scheduling, event bus, consensus,
    layered shared memory, the task-planning pipeline and the fleet size.
    """

    enabled: bool = True
    # Concurrency for the DAG scheduler worker pool.
    workers: int = 4
    # Default task timeout (seconds); None disables timeouts.
    task_timeout: float | None = None
    # Default number of retries per task.
    default_retries: int = 0
    # Consensus quorum: fraction (0..1) or absolute count.
    quorum: float | int = 0
    # Event bus retained history window.
    event_history: int = 1_024
    # Layered shared-memory region capacities.
    shared_memory_capacity: int = 1_024
    # Decompose high-complexity pipeline tasks into sub-graphs.
    decompose_tasks: bool = True
    # Agent heartbeat timeout in seconds.
    heartbeat_timeout: float = 30.0


@dataclass
class DesignLoopConfig:
    """Autonomous design-swarm loop (Pillar 5) configuration.

    Tunes the closed-loop stress -> reinforcement -> DFM -> cost workflow
    driven by the ``LeadArchitectAgent`` / ``FEAStressAgent`` /
    ``DFMManufacturingAgent`` / ``CostEstimatorAgent`` team.
    """

    enabled: bool = True
    # Maximum reinforcement/DFM cycles before the loop gives up.
    max_iterations: int = 10
    # Target factor of safety vs material yield (must be > 1.0).
    target_safety_factor: float = 1.5
    # Maximum cross-section growth per reinforcement step.
    reinforce_max_growth_per_step: float = 1.5
    # Let the loop switch to an alternative process when DFM fails.
    process_switching: bool = True
    # Default order quantity used by the cost estimator.
    default_quantity: int = 1
    # Default manufacturing process for the DFM gate.
    default_process: str = "machining"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------


@dataclass
class CADConfig:
    """
    Master configuration for CADGenesis-LM v2.0.

    Usage::

        cfg = CADConfig()
        cfg.model.d_model = 512           # override for smaller experiments
        cfg.save("my_experiment.json")

        cfg2 = CADConfig.load("my_experiment.json")
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    design: DesignLoopConfig = field(default_factory=DesignLoopConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Paths
    output_dir: str = "outputs/cadgenesis_v2"
    cache_dir: str = ".cache/cadgenesis"
    tokenizer_path: str | None = None

    # Experiment tracking
    experiment_name: str = "cadgenesis_v2_default"
    seed: int = 42

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Fast-fail on obviously invalid configurations."""
        m = self.model
        total_heads = (
            m.self_attn_heads
            + m.geometry_attn_heads
            + m.constraint_attn_heads
            + m.memory_attn_heads
            + m.agent_attn_heads
            + m.uncertainty_attn_heads
        )
        if total_heads != m.nhead:
            raise ValueError(
                f"Attention head counts sum to {total_heads} but nhead={m.nhead}. "
                "Adjust model.{{self,geometry,constraint,memory,agent,uncertainty}}_attn_heads."
            )

        if m.d_model % m.nhead != 0:
            raise ValueError(f"d_model={m.d_model} must be divisible by nhead={m.nhead}.")

        if m.agent_attn_heads > 0 and not m.use_multi_agent_system:
            raise ValueError(
                "agent_attn_heads > 0 requires use_multi_agent_system=True "
                "(agent states are produced by the multi-agent system)."
            )
        if m.memory_attn_heads > 0 and not m.use_memory_system:
            raise ValueError(
                "memory_attn_heads > 0 requires use_memory_system=True "
                "(memory K/V come from the layer-integrated memory pools)."
            )

        if m.use_moe and any(
            [m.num_experts < 1, m.top_k_experts < 1, m.top_k_experts > m.num_experts]
        ):
            raise ValueError(
                f"MoE requires 1 <= top_k_experts <= num_experts; "
                f"got num_experts={m.num_experts}, top_k_experts={m.top_k_experts}."
            )

        if m.num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0.")

        if m.rope_scaling_type not in ("none", "linear", "ntk", "yarn"):
            raise ValueError(
                f"rope_scaling_type must be one of 'none', 'linear', 'ntk', "
                f"'yarn'; got '{m.rope_scaling_type}'."
            )
        if m.rope_scaling_factor <= 0:
            raise ValueError("rope_scaling_factor must be > 0.")
        if m.use_ssm and m.ssm_every_n_blocks < 1:
            raise ValueError("ssm_every_n_blocks must be >= 1 when use_ssm=True.")
        if m.use_ssm and m.d_model % m.ssm_heads != 0:
            raise ValueError(f"d_model={m.d_model} must be divisible by ssm_heads={m.ssm_heads}.")

        if m.attention_backend not in ("math", "sdpa", "flash", "linear", "gqa", "mla"):
            raise ValueError(
                f"attention_backend must be one of 'math', 'sdpa', 'flash', "
                f"'linear', 'gqa', 'mla'; got '{m.attention_backend}'."
            )
        if m.attention_backend == "gqa" and m.num_kv_heads is not None:
            if m.num_kv_heads < 1 or m.num_kv_heads > m.self_attn_heads:
                raise ValueError(
                    f"num_kv_heads must satisfy 1 <= num_kv_heads <= self_attn_heads; "
                    f"got {m.num_kv_heads}."
                )
            if m.self_attn_heads % m.num_kv_heads != 0:
                raise ValueError(
                    f"self_attn_heads ({m.self_attn_heads}) must be divisible by "
                    f"num_kv_heads ({m.num_kv_heads})."
                )
        if m.attention_backend == "mla":
            mla_head_dim = m.d_model // m.self_attn_heads
            if m.qk_rope_head_dim >= mla_head_dim:
                # Small models (e.g. ``mini``: d_model=128, 2 heads → head_dim
                # 64) ship with the default qk_rope_head_dim=64, which exceeds
                # head_dim.  Auto-clamp to the largest *even* value below
                # head_dim (RoPE requires an even dimension; see
                # ``RotaryEmbedding``) instead of failing, so MLA works out of
                # the box on small configs (v6.1 §4.6).  The clamped value is
                # applied to every block via ``_block_kwargs`` at build time.
                max_even = mla_head_dim - 1 - ((mla_head_dim - 1) % 2)
                m.qk_rope_head_dim = max_even
            if m.qk_rope_head_dim < 1:
                raise ValueError(
                    f"MLA requires 1 <= qk_rope_head_dim < head_dim "
                    f"(d_model // self_attn_heads = {mla_head_dim}); "
                    f"got qk_rope_head_dim={m.qk_rope_head_dim}."
                )
            if m.kv_lora_rank < 1:
                raise ValueError(f"kv_lora_rank must be >= 1; got {m.kv_lora_rank}.")

        if m.interaction_heads < 1:
            raise ValueError(f"interaction_heads must be >= 1; got {m.interaction_heads}.")

        if m.sparse_attention and m.sparse_attention_pattern not in (
            "local",
            "global",
            "sliding_window",
            "block_sparse",
            "mixed",
        ):
            raise ValueError(
                f"sparse_attention_pattern must be one of 'local', 'global', "
                f"'sliding_window', 'block_sparse', 'mixed'; "
                f"got '{m.sparse_attention_pattern}'."
            )
        if m.sliding_window_size < 1 or m.local_attention_size < 1:
            raise ValueError("sliding_window_size and local_attention_size must be >= 1.")
        if m.num_global_tokens < 1 or m.block_size < 1:
            raise ValueError("num_global_tokens and block_size must be >= 1.")
        if m.use_multi_scale_attention and m.multi_scale_local_window < 1:
            raise ValueError("multi_scale_local_window must be >= 1.")
        if m.multi_scale_medium_window < m.multi_scale_local_window:
            raise ValueError("multi_scale_medium_window must be >= multi_scale_local_window.")
        if not (0.0 <= m.computation_budget <= 1.0):
            raise ValueError(f"computation_budget must be in [0, 1]; got {m.computation_budget}.")
        if m.use_specialized_moe and (
            m.experts_per_domain < 1
            or not (1 <= m.top_k_domain_experts <= 5 * m.experts_per_domain)
        ):
            raise ValueError(
                "specialized MoE requires experts_per_domain >= 1 and "
                "1 <= top_k_domain_experts <= 5 * experts_per_domain."
            )

        if self.training.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError(
                f"mixed_precision must be one of 'no', 'fp16', 'bf16'; "
                f"got '{self.training.mixed_precision}'."
            )

        if self.design.max_iterations < 1:
            raise ValueError(
                f"design.max_iterations must be >= 1; got {self.design.max_iterations}."
            )
        if self.design.target_safety_factor <= 1.0:
            raise ValueError(
                "design.target_safety_factor must be > 1.0; got "
                f"{self.design.target_safety_factor}."
            )
        if self.design.reinforce_max_growth_per_step <= 1.0:
            raise ValueError(
                "design.reinforce_max_growth_per_step must be > 1.0; got "
                f"{self.design.reinforce_max_growth_per_step}."
            )

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, raw: dict) -> CADConfig:
        """Rebuild a config from :meth:`to_dict` output (checkpoint round-trip)."""
        cfg = cls(
            tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            observability=ObservabilityConfig(**raw.get("observability", {})),
            multimodal=MultimodalConfig(**raw.get("multimodal", {})),
            world_model=WorldModelConfig(**raw.get("world_model", {})),
            agents=AgentsConfig(**raw.get("agents", {})),
            design=DesignLoopConfig(**raw.get("design", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )
        for key in ("output_dir", "cache_dir", "tokenizer_path", "experiment_name", "seed"):
            if key in raw:
                object.__setattr__(cfg, key, raw[key])
        cfg._validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> CADConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        cfg = cls(
            tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            observability=ObservabilityConfig(**raw.get("observability", {})),
            multimodal=MultimodalConfig(**raw.get("multimodal", {})),
            world_model=WorldModelConfig(**raw.get("world_model", {})),
            agents=AgentsConfig(**raw.get("agents", {})),
            design=DesignLoopConfig(**raw.get("design", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )
        # Scalar fields
        for key in ("output_dir", "cache_dir", "tokenizer_path", "experiment_name", "seed"):
            if key in raw:
                object.__setattr__(cfg, key, raw[key])
        cfg._validate()
        return cfg

    @classmethod
    def mini(cls) -> CADConfig:
        """
        Returns a small configuration suitable for single-GPU experimentation
        and unit tests � backward-compatible with the original CADGenesisMini.

        Unlike the lean default, ``mini`` keeps the *full research stack*
        (multi-agent system, memory pools, neuro-symbolic engine, RLAIF reward
        model) enabled so every subsystem stays exercised in the test suite.
        """
        cfg = cls(
            model=ModelConfig(
                d_model=128,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=256,
                self_attn_heads=2,
                geometry_attn_heads=1,
                constraint_attn_heads=0,
                memory_attn_heads=0,
                agent_attn_heads=1,
                uncertainty_attn_heads=0,
                use_multi_agent_system=True,
                use_memory_system=True,
                use_neuro_symbolic_reasoning=True,
                use_rlaf_reward_model=True,
            ),
            training=TrainingConfig(
                batch_size=64,
                max_epochs=8,
                gradient_checkpointing=False,
                mixed_precision="no",
                rlaf_reward_weight=0.01,
            ),
            tokenizer=TokenizerConfig(
                num_bins=20,
                lang_vocab_size=512,
                # 1024-slot mini vocabulary, matching
                # ``AutonomousCADTokenizer.build_mini()`` exactly: 512 CAD
                # slots (SPECIAL 64 + NUMERIC 384 + GEOMETRY 32 + FEATURE 32)
                # with the LANGUAGE family starting at id 512.  The model's
                # ``cad_vocab_size`` (slot sum) is then == the tokenizer's
                # language range start, so CAD ids (0..511) and language ids
                # (512..1023) can never collide (v6.1 §4.4 / §4.5).
                geometry_token_slots=32,
                feature_token_slots=32,
                constraint_token_slots=0,
                material_token_slots=0,
                assembly_token_slots=0,
                manufacturing_token_slots=0,
                simulation_token_slots=0,
                numeric_token_slots=384,
                special_token_slots=64,
            ),
        )
        # Patch head count to match nhead=4
        cfg.model.constraint_attn_heads = 0
        cfg.model.memory_attn_heads = 0
        cfg.model.uncertainty_attn_heads = 0
        return cfg

    @classmethod
    def from_preset(cls, name: str) -> CADConfig:
        """
        Lean, production-oriented scale ladder (encoder-decoder, standard
        self-attention + encoder-decoder cross-attention, no experimental
        subsystems):

        * ``nano``  — 128 dim / 3+3 layers (≈ ``mini`` but lean).
        * ``small`` — 384 dim / 6+6 layers (trains on a single CPU/GPU).
        * ``base``  — 768 dim / 12+12 layers, GQA.
        * ``1.5b``  — 1536 dim / 16+16 layers, GQA (≈1.54B params).
        * ``large`` — 1536 dim / 24+24 layers, GQA + MTP (≈2.28B params).
        * ``production`` — alias for ``large`` (see :meth:`production`).
        """
        ladder: dict[str, dict[str, Any]] = {
            "nano": dict(
                d_model=128,
                nhead=4,
                self_attn_heads=2,
                geometry_attn_heads=2,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=512,
            ),
            "small": dict(
                d_model=384,
                nhead=8,
                self_attn_heads=6,
                geometry_attn_heads=2,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=1536,
                attention_backend="gqa",
                num_kv_heads=2,
            ),
            "base": dict(
                d_model=768,
                nhead=12,
                self_attn_heads=8,
                geometry_attn_heads=4,
                num_encoder_layers=12,
                num_decoder_layers=12,
                dim_feedforward=3072,
                attention_backend="gqa",
                num_kv_heads=4,
            ),
            "1.5b": dict(
                d_model=1536,
                nhead=16,
                self_attn_heads=12,
                geometry_attn_heads=4,
                num_encoder_layers=16,
                num_decoder_layers=16,
                dim_feedforward=6144,
                attention_backend="gqa",
                num_kv_heads=4,
            ),
            "large": dict(
                d_model=1536,
                nhead=16,
                self_attn_heads=12,
                geometry_attn_heads=4,
                num_encoder_layers=24,
                num_decoder_layers=24,
                dim_feedforward=6144,
                attention_backend="gqa",
                num_kv_heads=4,
                mtp_depth=1,
                mtp_weight=0.1,
            ),
        }
        if name not in ladder:
            raise ValueError(
                f"unknown preset {name!r}; choose from {sorted([*ladder, 'production'])}"
            )
        return cls(model=ModelConfig(**ladder[name]))

    @classmethod
    def production(cls) -> CADConfig:
        """
        Production configuration: the ``large`` lean preset plus
        DeepSeek-V3-style sparse MoE FFNs and multi-token prediction.
        Requires a multi-GPU training environment (≈1B+ active params).
        """
        cfg = cls.from_preset("large")
        cfg.model.use_moe = True
        cfg.model.num_experts = 8
        cfg.model.top_k_experts = 2
        cfg.model.expert_dim = 512
        cfg.model.moe_aux_free_balancing = True
        cfg.model.moe_capacity_factor = 1.25
        cfg.model.moe_drop_tokens = True
        cfg.model.mtp_depth = 1
        cfg.training.mixed_precision = "bf16"
        return cfg
