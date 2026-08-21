# Pillar 1 — Foundation Model (CADGenesis-LM v6.0)

This document describes the Pillar 1 Foundation-Model upgrades delivered on top
of the existing `GeometryAwareTransformer` backbone. Everything is **additive and
backward compatible**: the baseline `GeometryAwareTransformer` forward contract
`forward(src, tgt, tgt_type, ...) -> (logits, confidence)` is preserved, every
new capability is off by default, and no working module was rewritten.

See `docs/m2_foundation_model.md` for the earlier completeness milestone and
`docs/v6_roadmap.md` for the roadmap.

## Capabilities delivered

| Capability | Module | Config gate |
| --- | --- | --- |
| Sparse attention (local / global / sliding-window / block-sparse / mixed) | `transformer/sparse_attention.py` | `model.sparse_attention` |
| Multi-scale attention (local + medium + global heads) | `transformer/multi_scale_attention.py` | `model.use_multi_scale_attention` |
| RoPE long-context scaling (linear interpolation / NTK-aware) | `transformer/positional.py` | `RotaryEmbedding(scaling_type=...)` |
| Specialised domain MoE (geometry / manufacturing / reasoning / simulation / optimization) | `transformer/specialized_moe.py` | `model.use_specialized_moe` |
| Hierarchical 5-stage transformer (Planner → Geometry → Constraint → Execution → Validation) | `transformer/hierarchical_transformer.py` | `model.use_hierarchical_transformer` |
| Dynamic computation routing (early exit + computation budgeting) | `transformer/dynamic_routing.py` | `model.computation_budget`, `model.early_exit_threshold` |
| Configurable transformer evolution framework (registry / plugins / versioning / experiments / builder) | `transformer/evolution/` | `model.evolution_plugins_enabled` |

## Architecture overview

```
                          ┌────────────────────────────────────────────────────┐
                          │              ConfigurationDrivenBuilder            │
                          │  spec {type: standard|hierarchical|stack} ────────┼───► GeometryAwareTransformer
                          │                                                  │───► HierarchicalCADTransformer
                          │  resolves "ffn" / "attention" via LayerRegistry   │───► RegistryStack
                          └───────────────────┬────────────────────────────────┘
                                              │ layer kinds
                                              ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │                      LayerRegistry                       │
                  │  rms_norm · swiglu_ffn · self_attention · *_attention    │
                  │  sparse_attention · multi_scale_attention · moe_ffn      │
                  │  specialized_moe_ffn · cad_block                         │
                  │  ▲ plugins register new kinds at runtime                 │
                  └─────────────────────────────────────────────────────────┘
```

## Class diagram

```
             nn.Module
                 │
     ┌───────────┼─────────────────────────────────────────────┐
     │           │                                             │
GeometryAwareTransformer                      HierarchicalCADTransformer
(unchanged)                                    │ 5 stages: planner(encoder)
                                               │ + geometry·constraint·execution·
                                               │   validation(decoder)
                                               │
                          ┌────────────────────┼─────────────────────┐
                          │                    │                     │
               DynamicRoutingController  SpecializedMoEFFN    CADTransformerBlock[]
                          │  ◄── uses        (injected via          │
                          │                    ffn/use_moe)         │
              ┌───────────┴───────────┐                            │
              │                       │                    MultiScaleAttention
    ComputationBudget        EarlyExitGate                SparseSelfAttention
```

Key components:

```
class Hierarchy:
    STAGE_NAMES = (planner, geometry, constraint, execution, validation)
    forward(src_ids, tgt_in_ids, tgt_type_ids, ...) -> (logits, confidence)
    encode(...)                          # planner stage
    aux_loss() -> Tensor                 # MoE load-balancing aux loss
    stage_report() -> dict               # per-stage depths + routing telemetry

class DynamicRoutingController:
    should_stop(step_idx, confidence=None) -> bool
    report() -> {exit_layer, exit_reason, layers_executed, savings_fraction, ...}

class SpecializedMoEFFN:
    router: Linear(E -> domains×experts_per_domain)
    experts: ModuleList[DomainExpert(domain, d_model, expert_dim)]
    forward(x) -> x'                    # top-k sparse activation across domains
    get_aux_loss() / domain_load() / add_domain(name)
```

## Sequence diagram — hierarchical forward with dynamic routing

```
 Trainer / InferenceEngine                HierarchicalCADTransformer          DynamicRoutingController
        │  forward(src, tgt, tgt_type)            │                                   │
        │──────────────────────────────────────────►                                  │
        │                                planner blocks (encode)                      │
        │                                for stage in (geometry, constraint,          │
        │                                  execution, validation):                    │
        │                                  for each block:                            │
        │                                    x, conf = block(x, ...)                  │
        │                                                  │ should_stop(step, conf)  │
        │                                                  │──────────────────────────►
        │                                                  │      True/False          │
        │                                                  │◄─────────────────────────│
        │                                    if True: break (skip remaining layers)   │
        │  (logits, confidence) ◄──────────────────────────│                          │
        │◄─────────────────────────────────────────────────┘                          │
```

## API surface

All names are re-exported from `cadgenesis.transformer` and
`cadgenesis.transformer.transformer`.

### Attention

| Symbol | Notes |
| --- | --- |
| `SparseSelfAttention(d_model, num_heads, pattern, ...)` | Contract matches `SelfAttention.forward(x, attn_mask=None, use_rope=True)`. |
| `build_sparse_attention(pattern, **kw)` | `pattern ∈ {local, global, sliding_window, block_sparse, mixed}`. |
| `sparse_attention_mask(pattern, seq_len, ...)` | Additive mask builder (used by `MultiScaleAttention`). |
| `SPARSE_PATTERNS`, `SparseAttentionPattern` | Enumerations. |
| `MultiScaleAttention(d_model, num_heads, head_fractions, ...)` | Parallel local/medium/global head groups; `scale_report`. |

### Positional

| Symbol | Notes |
| --- | --- |
| `RotaryEmbedding(dim, ..., scaling_factor=1.0, scaling_type="none")` | `scaling_type ∈ {none, linear, ntk}`. Default `none` is byte-for-byte identical to legacy. |

### Mixture of experts

| Symbol | Notes |
| --- | --- |
| `SpecializedMoEFFN(d_model, expert_types=DEFAULT_DOMAIN_EXPERTS, ...)` | Domain-labelled top-k MoE with Shazeer aux loss. |
| `DomainExpert(domain, d_model, expert_dim, dropout)` | Two-layer GELU expert. |
| `register_expert_type(name)` / `registered_expert_types()` | Runtime domain taxonomy extension. |
| `DEFAULT_DOMAIN_EXPERTS` | geometry, manufacturing, reasoning, simulation, optimization. |

### Dynamic routing

| Symbol | Notes |
| --- | --- |
| `ComputationBudget(budget)` | Layer-count cap, `max_layers(total)`. |
| `EarlyExitGate(threshold)` | Confidence-triggered early exit (0 = disabled). |
| `DynamicRoutingController(total_layers, budget, early_exit_threshold, min_steps)` | Decision loop + telemetry `report()`. |

### Hierarchical transformer

| Symbol | Notes |
| --- | --- |
| `HierarchicalCADTransformer(config)` | 5-stage encoder-decoder; same forward contract as `GeometryAwareTransformer`. |
| `STAGE_NAMES` | The five stage names. |
| `aux_loss()`, `stage_report()`, `total_decoder_layers` | Diagnostics / MoE training helpers. |

### Evolution framework

| Symbol | Notes |
| --- | --- |
| `LayerRegistry` / `global_registry` | Kind → factory; `register`, `build`, `register_from_module`, `unregister`. |
| `Plugin`, `PluginManager`, `register_layer` | Plugin system; layers register at runtime. |
| `ArchitectureVersion`, `VersionedArchitecture`, `hash_architecture` | Semantic versioning + canonical SHA-256 content hashes. |
| `ExperimentRecord`, `ExperimentRegistry` | Reproducible experiment store (in-memory + JSON). |
| `ConfigurationDrivenBuilder` | Builds standard / hierarchical / stack models from JSON specs. |
| `RegistryStack` | Sequential stack of registry-resolved layers. |

## Backward compatibility

- The default `CADConfig.mini()` and full configs are unchanged: all new fields
  default to off (`sparse_attention=False`, `use_hierarchical_transformer=False`,
  `use_specialized_moe=False`, `computation_budget=1.0`,
  `early_exit_threshold=0.0`, `scaling_type="none"`).
- `GeometryAwareTransformer` and `SelfDesigningTransformer` are untouched; all
  122 baseline transformer tests continue to pass (245 in the transformer suite).
- `CADInferenceEngine` consumes the hierarchical model through the existing
  duck-typed `(logits, confidence)` contract — no engine changes were required.

## Verification

```text
pytest tests/transformer -q          245 passed
pytest tests -q                      (full suite; see project report)
python benchmarks/foundation_benchmarks.py --section hierarchical
   [full  ] ~14.9ms  layers=4/4  savings=0%
   [budget] ~ 9.5ms  layers=2/4  savings=50%
python tools/profile_foundation.py --budget 0.5   (shows skipped execution/validation)
python scripts/audit_repo.py                     (Foundation Model pillar coverage)
```
