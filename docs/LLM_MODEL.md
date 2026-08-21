# CADGenesis-LM — The LLM Model, End to End

Complete technical reference for the CADGenesis-LM v6.1 language model:
tokenization → embeddings → encoder → decoder → loss → training → inference,
including every optional subsystem, the configuration space, and a changelog
of the verified bugs fixed in this audit.

---

## 1. What the model is

`GeometryAwareTransformer` (`src/cadgenesis/transformer/geometry_transformer.py`)
is a **bidirectional-encoder / causal-decoder transformer** that translates
*engineering language prompts* into *CAD program token sequences*.

- **Encoder input**: natural-language / engineering text token ids `(B, S)`.
- **Decoder input**: shifted-right CAD token ids `(B, T)` + type ids `(B, T)`.
- **Output**: CAD logits `(B, T, cad_vocab_size)` + optional confidence logits `(B, T, 1)`.
- **Loss**: masked cross-entropy over target tokens (+ optional confidence,
  MoE auxiliary, MTP, and RLAIF reward terms).
- **Inference**: `CADEngine` (`src/cadgenesis/inference/engine.py`) with
  greedy / beam / sampling, plus an exact KV-cached `decode_step`.

The default (`ModelConfig`) is deliberately **lean**: a plain attention
mixture with self-attention + geometry cross-attention only. Every
experimental subsystem (multi-agent, memory pools, neuro-symbolic rules,
RLAIF reward model, MoE, SSM, MTP, self-designing, sparse/multi-scale
attention, BitNet, …) is **off unless explicitly enabled**.

---

## 2. Configuration space

### 2.1 `ModelConfig` (defaults)

| Field | Default | Meaning |
|---|---|---|
| `d_model` | 1024 | Embedding width |
| `nhead` | 16 | Total attention heads (must equal the head-type sum) |
| `num_encoder_layers` / `num_decoder_layers` | 12 / 12 | Block counts |
| `dim_feedforward` | 4096 | SwiGLU hidden width |
| `dropout` | 0.1 | Dropout (attention + FFN) |
| `max_seq_len` | 2048 | Initial sinusoidal table size (grown on demand — no hard cap since v6.1) |
| `rope_theta` / `rope_scaling_type` / `rope_scaling_factor` | 10000 / none / 1.0 | RoPE base & long-context scaling (linear / ntk / yarn) |
| `max_position_embeddings` | 2048 | RoPE precompute cap; the model effectively raises it to `max(max_position_embeddings, max_seq_len)` so the configured context is always covered |
| `self_attn_heads` / `geometry_attn_heads` | 8 / 8 | Standard heads |
| `constraint_attn_heads` / `memory_attn_heads` / `agent_attn_heads` / `uncertainty_attn_heads` | 0 | Specialized heads (forced 0 unless subsystem flag on) |
| `use_multi_agent_system` / `use_memory_system` / `use_neuro_symbolic_reasoning` / `use_rlaf_reward_model` | False | Subsystem switches |
| `use_confidence_head` | True | Per-token confidence head |
| `use_ssm` / `ssm_every_n_blocks` / `ssm_heads` | False / 3 / 4 | Gated DeltaNet interleave |
| `use_moe` / `num_experts` / `top_k_experts` / `expert_dim` | False / 4 / 2 / None | Sparse MoE FFN |
| `num_shared_experts` / `shared_expert_dim` | 0 / None | DeepSeek-V3 shared expert |
| `moe_aux_free_balancing` / `moe_balance_speed` / `moe_z_loss_weight` | False / 0.001 / 1e-3 | Aux-loss-free MoE |
| `moe_capacity_factor` / `moe_drop_tokens` | None / False | Expert capacity / dropping |
| `attention_backend` | `"math"` | One of `math`, `sdpa`, `flash`, `linear`, `gqa`, `mla` |
| `num_kv_heads` / `kv_lora_rank` / `qk_rope_head_dim` | None / 64 / 64 | GQA / MLA options |
| `mtp_depth` / `mtp_weight` | 0 / 0.1 | Multi-token prediction head |
| `geometry_pos_encoding` | False | Learned X/Y/Z coordinate encoding |
| `feature_interaction` / `interaction_heads` | False / 2 | Gated cross-feature sub-layer |
| `sparse_attention` / `use_multi_scale_attention` / `use_hierarchical_transformer` / `use_specialized_moe` | False | Alternate architectures |
| `early_exit_threshold` / `computation_budget` | 0.0 / 1.0 | Dynamic routing |

Validation (`_validate`): head counts must sum to `nhead`; `ssm` requires
`ssm_every_n_blocks >= 1` and `d_model % ssm_heads == 0`; GQA requires
`num_heads % num_kv_heads == 0`; **MLA auto-clamps** `qk_rope_head_dim` to the
largest even value below `head_dim` when it exceeds it (v6.1 §4.6) instead of
failing, so small models (e.g. mini, head_dim 64) work with MLA out of the
box. The same clamp is re-applied at model build time (`_block_kwargs`) for
configs mutated after construction.

### 2.2 `TokenizerConfig` — vocabulary slot budget

| Family | Default slots | Description |
|---|---|---|
| `special_token_slots` | 64 | `<pad>`, `<bos>`, `<eos>`, `<sep>`, `<mask>`, … |
| `numeric_token_slots` | 1024 | Quantized numeric parameters |
| `geometry_token_slots` | 512 | Primitive + B-Rep tokens |
| `feature_token_slots` | 512 | CAD feature operations |
| `constraint_token_slots` | 256 | Parametric constraints |
| `material_token_slots` | 256 | Materials |
| `assembly_token_slots` | 256 | Assembly relations |
| `manufacturing_token_slots` | 256 | Manufacturing processes |
| `simulation_token_slots` | 256 | Simulation / physics |
| `lang_vocab_size` | 32000 | Text-vocab size for the encoder embedding |

Default `cad_vocab_size` = **3392** (sum of CAD families + specials).
Default total flat vocab = 3392 + 32000 = **35392**.

### 2.3 `TrainingConfig` (defaults)

`batch_size=64`, `grad_accum_steps=4`, `max_epochs=100`, `warmup_steps=2000`,
`lr=3e-4`, `weight_decay=0.01`, `max_grad_norm=1.0`,
`gradient_checkpointing=True`, `mixed_precision="bf16"`,
`schedule="cosine"` (+ WSD options), `use_packing=False` (+ packed length caps),
`label_smoothing=0.0`, `moe_aux_scale=0.01`, `confidence_loss_weight=0.1`,
`rlaf_reward_weight=0.0` (RLAIF reward-maximisation; 0 = off).

### 2.4 Presets (`CADConfig.from_preset`)

| Preset | d_model | Heads (self/geom) | Layers | Backend | Notes |
|---|---|---|---|---|---|
| `nano` | 128 | 2/2 | 3+3 | math | lean mini |
| `small` | 384 | 6/2 | 6+6 | **gqa** (kv=2) | CPU/GPU |
| `base` | 768 | 8/4 | 12+12 | **gqa** (kv=4) | |
| `1.5b` | 1536 | 12/4 | 16+16 | **gqa** (kv=4) | ≈1.54B params |
| `large` | 1536 | 12/4 | 24+24 | **gqa** (kv=4) | + MTP ≈2.28B |
| `production` | = large | | | | + MoE (8 experts, top-2), aux-free balancing, capacity 1.25, drop tokens, bf16 |

`CADConfig.mini()`: 128-dim / 3+3 layers with the **full research stack
enabled** (agents, memory, neuro-symbolic, RLAIF) so every subsystem is
exercised in tests; `lang_vocab_size=512`, `rlaf_reward_weight=0.01`.  Since
v6.1 the mini config uses the **1024-slot vocabulary layout** (see §3.1):
512 CAD slots (SPECIAL 64 + NUMERIC 384 + GEOMETRY 32 + FEATURE 32) so the
model's `cad_vocab_size` is exactly 512 — matching the mini tokenizer's
language range start instead of over-allocating 3392.

---

## 3. Tokenization

### 3.1 Flat vocabulary (`CADVocabulary`)

Contiguous ID ranges in registry order: SPECIAL → NUMERIC → GEOMETRY →
FEATURE → CONSTRAINT → MATERIAL → ASSEMBLY → MANUFACTURING → SIMULATION →
LANGUAGE. Each `VocabularyRecord` carries `token_id`, `type_id` (token
family 0..9), and a canonical text form (TOON — "Token ON", the canonical
human-readable serialization; **no JSON replacement**).

- `tokenize` → numeric→token (`numeric_tokenize`, bins: `num_bins=256`,
  `param_min=0`, `param_max=1000`, `angle_bins=360`), feature/geometry →
  structured token strings.
- `encode_cad_sequence` / `decode_cad_sequence`: flat-sequence round-trip.
- `remap_ids` / `type_id_of`: family-aware id mapping.
- **Mini layout** (`build_mini`): SPECIAL 0–63, NUMERIC 64–447, GEOMETRY
  448–479, FEATURE 480–511, LANGUAGE 512–1023 (total 1024 slots, 412
  registered).  Since v6.1 the mini *model* allocates a matching
  `cad_vocab_size=512` (was 3392 over-allocation).
- **Separation invariant (v6.1 §4.5)**: the LANGUAGE family always starts at
  the model's `cad_vocab_size` — 3392 (default) and 512 (mini) — so CAD ids
  and language ids occupy disjoint, adjacent ranges and can never collide.
  Covered by `tests/tokenizer/test_vocab_separation.py`.

### 3.2 Text tokenizer (`AutonomousCADTokenizer`)

The encoder consumes **text** ids through a separate `lang_embed`
(`lang_vocab_size` wide):

- **Legacy default**: `LegacyWordTokenizer` — simple word → id starting at 0
  (used by `build`/`build_mini`).
- **BPE option**: HF-style `BPETokenizer` (`lang_vocab_size=32000`).
- Type ids for CAD targets are derived from the flat vocab via a
  `token_id → type_id` table.

> Known inconsistency (documented, not a crash): legacy text ids start at 0,
> overlapping the CAD id space in the *flat registry* (e.g. `type_id_of(300)`
> may resolve a text token as a geometry token). The model never mixes the
> two embedding tables, so training is unaffected; cross-vocab lookups on
> mixed sequences should go through the dedicated tokenizers.

---

## 4. Architecture (forward path)

```
src_ids (B,S) ── lang_embed ──┐
                              ├─ SinusoidalPositionalEncoding (+optional GeometryPositionalEncoding)
                              └─ (*sqrt(d_model))
        ┌────────────────── ENCODER (bidirectional) ──────────────────┐
        │ ×L  CADTransformerBlock:                                    │
        │    norm1 → MultiHeadAttentionMixture → residual             │
        │    [optional FeatureInteractionLayer]                       │
        │    norm2 → SwiGLU | SparseMoE | SpecializedMoE → residual   │
        │    [optional GatedDeltaNet after block]                    │
        └─────────────────────────────────────────────────────────────┘
                              ↓ encoder_hidden_states (B,S,C)

tgt_ids (B,T) ── cad_embed + type_embed ── pos_enc ─┐
        ┌────────────────── DECODER (causal) ─────────────────────────┐
        │ ×L  same block; geometry head cross-attends encoder states  │
        │    (masked by cross_attn_mask / padding)                    │
        └─────────────────────────────────────────────────────────────┘
   decoder_norm → out_proj (= cad_embed weight, tied) → logits (B,T,V)
                → confidence_head → confidence (B,T,1)   [optional]
                → reward_model → last_reward (B,1)       [optional]
                → mtp_head(hidden) → MTP logits          [optional]
```

### 4.1 Embeddings (`embeddings.py`)

- `lang_embed`: text ids, `padding_idx=0`.
- `cad_embed`: CAD ids, `padding_idx=0`; **weight-tied** with `out_proj` (LM head).
- `type_embed`: 10 token-family type embeddings added to decoder inputs.
- `_init_weights`: all 2D+ parameters initialized `N(0, 0.02)`.

### 4.2 Positional encodings (`positional.py`)

- `SinusoidalPositionalEncoding`: standard sines/cosines.  The table is
  **grown on demand** (doubling) when a sequence exceeds the initial
  `max_seq_len` (v6.1 §4.7) — the legacy hard cap (ValueError beyond 2048)
  is gone, so long-context configs work without raising.
- `RotaryEmbedding`: applied inside attention backends; supports
  `linear` / `ntk` / `yarn` long-context scaling. Class-level defaults are
  reset to canonical values at each model build (fixes cross-model leakage).
  Requires an **even** `dim` (fail-fast guard); sequences beyond the
  precomputed table are computed on the fly, so RoPE itself never caps the
  context.
- `GeometryPositionalEncoding` (optional): adds learned X/Y/Z coordinate
  encodings when `geometry_pos_encoding=True`.

### 4.3 `CADTransformerBlock` (`transformer_block.py`)

Pre-norm RMSNorm; attention mixture with residual; optional gated
`FeatureInteractionLayer`; SwiGLU (or MoE) FFN with residual; optional
per-token `layer_gate` (self-designing routing) and `head_weights`
(adaptive head modulation). `forward_cached` mirrors `forward` for KV-cached
decoding; **cached ≡ full** is guaranteed (verified, see §10).

### 4.4 Attention mixture (`attention.py`)

`MultiHeadAttentionMixture` runs up to six parallel heads, weighted by a
learned per-token gate `softmax(Linear(x))` over active heads:

| Head | Keys | Notes |
|---|---|---|
| `SelfAttention` (or backend) | self | backend-swappable |
| `GeometryAttention` | encoder states | fixed scale `geom_scale=1.0` (learnable scale was frozen — collapsed in practice) |
| `ConstraintAttention` | — | uses an explicit constraint mask |
| `MemoryAttention` | memory bank | subsystem |
| `AgentAttention` | agent states | subsystem |
| `UncertaintyAttention` | self | also emits confidence logits |

### 4.5 Attention backends (`efficient_attention.py`, `modern_attention.py`)

| Backend | Module | Complexity | Notes |
|---|---|---|---|
| `math` | `SelfAttention` | O(T²) | default; manual softmax |
| `sdpa` | `SDPASelfAttention` | O(T²) fused | `F.scaled_dot_product_attention` |
| `flash` | `SDPASelfAttention` | fused | engages flash kernel on supported GPUs |
| `linear` | `LinearAttention` | O(T·N) | Performer FAVOR+ random features; causal via cumsum; restrictive masks → exact quadratic fallback |
| `gqa` | `GroupedQueryAttention` | O(T²) grouped | `num_kv_heads` KV heads |
| `mla` | `MultiHeadLatentAttention` | latent KV | DeepSeek-V3 latent `c_KV`, nope+rope key split |

**Mask contract (all backends, after the audit fix):** `attn_mask=None` ⇒
bidirectional (encoder); a triangular `-inf` mask ⇒ causal (decoder);
block-diagonal masks ⇒ packing; the encoder is always bidirectional.

**Dead-row safety (v6.1 §4.1):** `safe_softmax` (in `attention.py`) zeros
fully-masked score rows after softmax, and `repair_fully_masked_rows`
re-opens a diagonal self-slot for dead query rows of square masks (and the
whole row for cross masks) before attention.  `encode`/`decode` apply it
after merging padding masks into packed block-diagonal masks, so
all-(-inf) score rows can never produce `0/0` softmax NaNs.  The
`LinearAttention` masked fallback carries the same guard.

### 4.6 FFNs

- `SwiGLU` (default): `w2(SiLU(w1 x) * w3 x)`.
- `SparseMoEFFN` (`moe.py`): top-k routing with jitter, optional
  aux-free balancing (DeepSeek-V3 expert bias), router **z-loss**,
  capacity factor + token dropping, shared expert, expert add/remove
  (growable), and an **auxiliary load-balancing loss** that now flows into
  the training loss (§7).
- `SpecializedMoEFFN`: domain experts (geometry / manufacturing / reasoning /
  simulation / optimization).

### 4.7 Hybrid SSM (`ssm.py`)

`GatedDeltaNet` interleaves after every `ssm_every_n_blocks`-th block when
`use_ssm=True`; recurrent element-wise delta rule, `forward_cached` maintains
a per-sequence state.  Since v6.1 `forward_cached` is **exactly equivalent to
`forward` on the last step** (same projections, decay, dropout and output
projection) and participates in autograd (the `@torch.no_grad` decorator is
gone; callers wrap inference loops themselves) — verified for eval mode and
train mode with dropout off, and gradients flow through the recurrence
(`tests/transformer/test_ssm.py`). (The module docstring advertises an
outer-product state update; the implementation is a diagonal/element-wise
recurrence — internally consistent cached vs full.)

### 4.8 Multi-token prediction (`mtp.py`)

`MultiTokenPredictionHead` (DeepSeek-V3 style) predicts the next `mtp_depth`
tokens from the final hidden states, weight-tied logits; `mtp_loss`
aggregates per-depth masked CE, weighted by `mtp_weight` in training.

### 4.9 Output heads (`heads.py`)

`LMHead` (tied) + `ConfidenceHead`. `OutputHeads` bundles both.

---

## 5. Optional subsystems

### 5.1 Layer-Integrated Memory (`memory/memory_pools.py`)

`LayerIntegratedMemorySystem` — working/session/user/project/CAD/engineering/
manufacturing/simulation pools, combined memory bank, `refine` after every
block, retrieval top-k (config `MemoryConfig`). In the cached path the bank
is frozen for the whole generation.

### 5.2 Multi-Agent System (`agents/multi_agent_system.py`)

8 internal roles (Planner, Geometry, Constraint, Material, Assembly,
Manufacturing, Simulation, Validation) exchanging state over a shared bus;
output `(B, T, C)` consumed by `AgentAttention`.

### 5.3 Neuro-Symbolic Engine

`evaluate_constraints` refines decoder states with symbolic CAD rules
(validity, manufacturability, safety factor ≥ 1.5).

### 5.4 RLAIF Reward Model (`alignment/constitutional_ai.py`)

`RLAIFRewardModel`: mean-pooled hidden → 2-layer head → `tanh` score `(B,1)`.
Now **live**: `decode()`/`decode_step()` store `last_reward`; the trainer adds
`-rlaf_reward_weight · reward.mean()` (weight 0 = off; 0.01 in `mini()`).

### 5.5 Self-Designing Transformer (`transformer/self_designing/`)

`SelfDesigningTransformer` wraps the backbone with:

- `LayerRoutingController` — per-token, per-layer gates (`layer_gate`).
- `AdaptiveHeadsController` — per-token head modulation (`head_weights`).
- `LayerPruningController` + `AutomaticRollback` — prune / restore layers.
- `NeuralArchitectureSearch` — NAS over depths/widths; `apply_architecture`
  copies compatible weights.
- `ComplexityEvaluator` — prompt-complexity score.
- Expert growth / retirement on MoE blocks.
- Router/head masks are cached **per tensor identity** and cleared every
  forward (fix: previously keyed by sequence length, mixing encoder and
  decoder masks).

### 5.6 Other alternates

Sparse attention (local/global/sliding/block-sparse/mixed), multi-scale
attention (local+medium+global), hierarchical transformer (5-stage),
BitNet b1.58 quantization (`quantization/bitnet.py`), dynamic computation
routing (early exit / budget), transformer evolution framework
(`evolution/`), LoRA/QLoRA PEFT (`adapters/`).

---

## 6. Losses (`losses.py`)

| Term | Module | Definition |
|---|---|---|
| CE | `MaskedCrossEntropyLoss` | masked (pad) cross-entropy, optional label smoothing |
| Confidence | `ConfidenceLoss` | BCE-with-logits against per-token correctness (argmax == target), masked |
| MoE aux | — | `moe_aux_scale × Σ blocks aux_loss` (load-balancing or z-loss, live gradients) |
| MTP | `mtp.mtp_loss` | per-depth masked CE × `mtp_weight` |
| RLAIF | — | `-rlaf_reward_weight × reward.mean()` |

`CADSequenceLoss.forward(logits, targets, confidence_logits, target_confidence,
mask, aux_loss)` returns `(total, breakdown)` with keys `ce`, `confidence`,
`moe_aux`, `total`.

---

## 7. Training (`training/trainer.py`)

`CADTrainer`:

- **Two epoch loops**: `train_epoch` (padded) and `train_packed_epoch`
  (packed sequences with block-diagonal masks + `loss_mask`); validation
  mirrors both. Both now pass `target_confidence`, `aux_loss`, and the RLAIF
  reward term to the loss (fixes: confidence head and MoE aux loss were dead).
- **Batch prep**: ids → device, right-shift targets, `_map_type_ids`
  vectorized via a prebuilt `token_id → type_id` table (O(T) per batch,
  was O(V·T)), padding masks computed.
- **Grad accumulation** (`grad_accum_steps`), gradient clipping, AMP
  (`fp16`/`bf16` via autocast + GradScaler), DDP/FSDP awareness
  (`_model_attr`), DeepSpeed integration (`deepspeed.py`).
- **Scheduler**: cosine warmup/decay or WSD (`schedule` config);
  `configure_scheduler` divides by grad-accum and multiplies by `max_epochs`
  (caller must pass per-epoch steps).
- **Activation checkpointing (v6.1 §4.2)**: `training.gradient_checkpointing`
  is now applied.  `GeometryAwareTransformer._maybe_checkpoint` wraps each
  transformer block (and interleaved SSM layer) in
  `torch.utils.checkpoint.checkpoint(use_reentrant=False)` when the flag is
  set *and* the model is in training mode; evaluation/inference pass through
  untouched.  Verified numerically identical to the plain forward and
  identical gradients with dropout disabled (`test_training_stability.py`).
- **Checkpointing**: model/optimizer/scheduler/scaler state + config +
  dataset digest; resume replays; μ-Transfer support
  (`mu_transfer.py`), callbacks, metrics, profiler.

---

## 8. Inference (`inference/engine.py`, `geometry_transformer.py`)

- `CADEngine.generate`: `greedy`, `beam`, `sample`
  (temperature/top-k/top-p), with EOS handling; `generate_cached` variants
  use the KV cache.
- **Beam search (v6.1 §4.8)**: since v6.1 `beam()` implements
  * EOS handling — beams that emit EOS are retired and never expanded; the
    search stops after `beam_width` finished hypotheses or `max_len` tokens;
  * length normalization — hypotheses are ranked by
    `score / ((5 + len) / 6) ** length_penalty` (GNMT-style; default 0.6,
    0 = legacy plain cumulative log-probability);
  * score normalization — the final pick is the best *normalized* hypothesis
    across the finished set and the best unfinished beam.
  Covered by `tests/inference/test_beam.py`.
- `prepare_decoder_cache(src_ids)`: precomputes encoder states, **per-layer
  projected geometry/memory cross-attention KV**, frozen memory bank,
  per-block KV slots, SSM states, `position_offset`.
- `decode_step(token, type, cache)`: single-token autoregressive step. It now
  **seeds each block's first step with the precomputed cross-attention KV**
  (fix: previously the projection ran once per token), advances per-block
  self-attention KV and SSM states. `causal_mask=None` is correct here
  because `forward_cached` uses `is_causal=False` + KV-cache causality.
- **Guarantee**: cached decoding is exactly equivalent to full-sequence
  decoding (verified: max diff ≈ 5e-7 for a full subsystem-enabled model).
- `load_engine`/`save_engine` (CLI `generate.py`) wrap checkpoint + tokenizer
  + model for serving.

---

## 9. Verification

- Unit/contract tests: `tests/transformer`, `tests/inference`, `tests/tokenizer`,
  `tests/ir`, `tests/training`, `tests/memory`, `tests/agents`, … Full suite:
  **2477 tests — all pass** (baseline 2445 passed / 9 failed; the 9 failures
  were the training-NaN cluster, fixed by §4.1; full v6.1 suite re-run: 2477 passed in 231s)
- New v6.1 regression tests (23):
  `tests/training/test_training_stability.py` (3: packed-step finiteness,
  loss decrease, checkpointing ≡ plain forward),
  `tests/tokenizer/test_vocab_separation.py` (5: mini+default language-range
  invariant ×2, no CAD token in language range ×2, 1024-slot mini layout),
  `tests/config/test_mla_small_models.py` (4),
  `tests/transformer/test_long_context.py` (4),
  `tests/transformer/test_ssm.py` (+2: autograd + train-mode equivalence),
  `tests/inference/test_beam.py` (5: EOS termination, normalized-vs-raw
  ranking, unfinished-vs-finished pick, beam-width validation, finite-score
  filtering).
- Long-context (v6.1 §4.7): full forward with S=T=4100 on a mini model is
  finite (past the legacy 2048 hard cap).
- Independent runtime verification (this audit): encoder bidirectionality for
  every backend, MoE aux-loss gradient flow, linear-backend mask handling,
  padding masking, precomputed-KV consumption, cached≡full equivalence,
  reward-model liveness — 14/14 checks pass.

---

## 10. Bug-fix changelog (this audit)

| # | Bug | Fix | Verification |
|---|---|---|---|
| A1 | Encoder became **causal** with `gqa`/`sdpa`/`flash`/`mla` backends (`is_causal = attn_mask is None`) — every preset uses `gqa`, so all preset models had a broken encoder | Backends now default to `is_causal=False`; decoder always passes an explicit triangular mask; linear backend gained a bidirectional (total-sum) path | pos-0 output now changes with later tokens for all 6 backends |
| A2 | MoE aux/z-loss **detached** and never added to the training loss | Keep live tensors in `_aux_loss`; add model-level `aux_loss()` aggregate; trainer passes `aux_loss=` at all 3 sites; aux loss counts each token once (top-1, Switch formulation) | `autograd.grad(aux, params)` → 148,997 gradient elements |
| A3 | `_is_causal` always False (zeroed mask) → LinearAttention ignored all masks; broken dead `keep` broadcast | Correct `-inf` triangular detection; causal via cumsum, bidirectional via total sum, restrictive masks via exact quadratic fallback | packing-mask and causal-mask forwards finite |
| B1 | `src/tgt_key_padding_mask` accepted but **never applied** | Merge padding into self-attention (+ cross-attention) additive masks in `encode`/`decode` | padded vs unpadded outputs differ; packed forward finite |
| B2 | Confidence head **dead in training** (`target_confidence` never passed) | Trainer computes detached per-token correctness targets and passes them at all 3 sites | loss breakdown now includes `confidence` |
| B3 | `prepare_decoder_cache` cross-attn KV was **never consumed** | `decode_step` seeds first-step per-block KV with precomputed geometry/memory KV | poisoned precomputed KV now changes logits (diff 1.19) |
| B4 | Self-designing route/head caches keyed by **sequence length only** | Key by tensor identity (cache cleared per forward) | — |
| B5 | RLAIF `reward_model` was **dead code** | `decode`/`decode_step` compute `last_reward`; trainer adds `-rlaf_reward_weight·mean(reward)` (new config field, 0.01 in `mini()`) | `last_reward` populated; reward term live |
| B6 | `_map_type_ids` was **O(V·T)** per batch | Prebuilt type table, single gather — O(T) | — |
| B7 | `RotaryEmbedding` class-level defaults leaked **across models** | Reset to canonical defaults at every model build before applying its config | — |
| C1 | **Training NaN** (`train=nan val=nan` in train_script / fsdp / mu-transfer / distillation) | Root cause: `encode`/`decode` merged padding masks into packed block-diagonal masks by addition, re-killing dead rows → all-(-inf) score rows → softmax 0/0 NaN. Fixed with `safe_softmax` + `repair_fully_masked_rows` (attention.py, applied to all six attention classes incl. `forward_cached`, the LinearAttention masked fallback, and both mask-merge sites in the model) | all 9 previously-failing tests pass; packed mini training steps produce finite loss (≈8.2) and zero non-finite gradients |
| C2 | `gradient_checkpointing` flag declared but never applied | `_maybe_checkpoint` wraps each block + SSM layer (`use_reentrant=False`) when enabled and in train mode | loss & gradients identical to plain forward (dropout off); eval untouched |
| C3 | `GatedDeltaNet.forward_cached` was `@torch.no_grad` and skipped dropout → `forward` ≠ `forward_cached` in train mode | Removed the decorator, added the matching dropout; docstring states the equivalence contract | cached replay == full recurrence in eval *and* train mode (dropout off); grads flow through the recurrence |
| C4 | `CADConfig.mini()` `cad_vocab_size=3392` vs the 1024-slot mini vocab | mini tokenizer slots now match `build_mini()` exactly: SPECIAL 64 + NUMERIC 384 + GEOMETRY 32 + FEATURE 32 = 512 CAD ids, language starts at 512 | model `cad_vocab_size` == tokenizer language-range start == 512 |
| C5 | Text/CAD vocab separation not enforced | LANGUAGE family start must equal `cad_vocab_size` (both layouts); invariant + no-cross-token tests | `tests/tokenizer/test_vocab_separation.py` (5 tests) |
| C6 | MLA validation rejected small models (`qk_rope_head_dim=64` ≥ head_dim 64) | Auto-clamp to the largest **even** value below head_dim in `_validate` and `_block_kwargs`; `RotaryEmbedding` gained an even-dim fail-fast guard | MLA mini forward finite; `tests/config/test_mla_small_models.py` (4 tests) |
| C7 | `SinusoidalPositionalEncoding` hard-capped context at `max_seq_len` (2048) | Table grows on demand (doubling); RoPE precompute covers `max(max_position_embeddings, max_seq_len)` | 4100-token forward finite; `tests/transformer/test_long_context.py` (4 tests) |
| C8 | Beam search: no length normalization, EOS beams re-expanded, final pick by raw score only | Retire EOS beams, GNMT length penalty (`alpha=0.6`), best-normalized pick across finished + unfinished | scripted-model tests verify EOS termination and normalization |

---

## 11. Known issues (not fixed here)

1. `GatedDeltaNet` docstring promises `k⊗v` outer-product state; the
   implementation uses an element-wise (diagonal) recurrence (cached/full are
   mutually consistent since v6.1) and `forward` is a sequential Python loop
   (O(T) steps — no chunked/parallel scan yet).
2. Legacy text ids start at 0, overlapping the CAD id space in the *flat
   registry* (see §3.2). The flat-vocab CAD/language ranges themselves are
   disjoint by construction (v6.1 §4.5); only the legacy `LegacyWordTokenizer`
   ids reuse [0, n) in the *language embedding* space, which is a separate
   table — training is unaffected, but mixed-sequence lookups must go through
   the dedicated tokenizers.
3. `RotaryEmbedding` requires an even dimension (fail-fast guard). The MLA
   clamp always picks even values, but a hand-set odd `qk_rope_head_dim`
   raises at construction.
4. Scheduler step accounting assumes `configure_scheduler` receives
   per-epoch steps.
5. Environment: GPU is a GTX 1650 4 GB (the `v6_roadmap` targets RTX 3050
   8 GB); `.venv` is broken (points at a deleted Python 3.11) — use system
   `py` (3.14.7, torch 2.13.0+cu126). The v6.2 `HardwareAwareRuntime` must
   carry presets for both.