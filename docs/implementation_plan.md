# CADGenesis-LM v2.0 � Implementation Plan and Training Readiness

## 1. Current Status

### 1.1 Summary
CADGenesis-LM is now ready to begin model training with the full package training infrastructure in place. The repository now supports:

- `src/cadgenesis/cli/train.py` as the primary package training entrypoint
- `scripts/train.py` root wrapper forwarding into the package entrypoint
- `src/cadgenesis/training/trainer.py` with optimizer, scheduler, mixed precision support, gradient accumulation, checkpoint save/load, and validation
- Full CADGenesis v2 architecture selection via `--model-size full`
- Tokenizer and configuration synchronization for the full architecture path
- A mini experiment mode based on `CADConfig.mini()` for quick verification and low-resource testing

### 1.2 Verified Readiness
The training workflow has been verified end-to-end for both mini and full modes on synthetic data.

- `python -m cadgenesis.train --model-size full --epochs 1 --batch-size 2 --train-size 50 --valid-size 10 --output-dir outputs/test_full_arch` (full architecture verified)
- `python -m cadgenesis.train --epochs 1 --batch-size 8 --train-size 40 --valid-size 10 --output-dir outputs/test_device --device cuda`
- `python scripts/train.py --epochs 1 --batch-size 4 --train-size 20 --valid-size 5 --output-dir outputs/test_root`

All runs executed successfully and produced saved checkpoint files.

## 2. What Was Completed

### 2.1 Training Infrastructure
- Added a package-level CLI entrypoint in `src/cadgenesis/cli/train.py`
- Added `CADTrainer` with training/validation loops, gradient accumulation, mixed precision support, and learning-rate scheduler support
- Implemented checkpoint serialization for model, optimizer, scheduler, and AMP scaler state
- Added checkpoint resume support and best-validation checkpointing
- Verified device fallback from CUDA to CPU

### 2.2 Package Architecture
- Preserved the `src/cadgenesis/` package structure and core module layout
- Added package-level training entrypoint while keeping the `scripts/train.py` wrapper
- Kept the backward-compatible legacy dataset generation interface via `build_dataset` shim

### 2.3 Documentation
- Updated `README.md` with quick start examples and checkpoint resume usage
- Updated this implementation plan to reflect the actual completed state

## 3. Remaining Work Before Full Production Training

The repository is ready for initial training on synthetic data, but the following items remain important before scaling to production-quality training:

- Expand the tokenizer beyond the mini legacy token set to a full CAD-aware vocabulary and BPE-compatible language tokenizer
- Add experiment logging and metrics tracking (TensorBoard, WandB, or equivalent)
- Add a formal configuration / experiment management workflow beyond CLI overrides
- Solidify a dataset pipeline for real CAD prompts and higher-fidelity geometry sequences
- Validate the GeometryAwareTransformer and memory/agent subsystems on larger sequence lengths
- Add `save_every_n_steps` and `eval_every_n_steps` step-level checkpointing if training runs become long-lived
- Consider adding fault-tolerant resume/retry orchestration for interrupted GPU jobs

## 4. Recommended Next Steps for Training Phase

1. Start with the mini-verification infrastructure:
   - `python -m cadgenesis.train --epochs 10 --batch-size 32 --train-size 1000 --valid-size 200 --output-dir outputs/cadgenesis_train`
2. Use `--resume-from` when continuing from a saved checkpoint:
   - `python -m cadgenesis.train --resume-from outputs/cadgenesis_train/best_checkpoint.pt`
3. Monitor validation loss and inspect `outputs/cadgenesis_train/best_checkpoint.pt` for the best model state
4. Once the mini training loop is stable, expand the dataset and tokenizer before scaling to GPU training

## 5. Current Implementation Scope

This plan focuses on training readiness and does not yet claim full deployment-level CAD intelligence. The current implementation is sufficient to start training the model with the existing synthetic CAD dataset in both mini and full architecture modes.

## 6. v2.0 Completed Work (Self-Designing + Autonomous Tokenizer + Memory)

The following subsystems were implemented and are covered by the test suite (218 tests, all passing):

### 6.1 Self-Designing Transformer (`src/cadgenesis/transformer/self_designing/`)
- `architecture.py` — `ArchitectureSpec` (validated, serializable to `ModelConfig`),
  `ArchitectureSearchSpace`, `NeuralArchitectureSearch` (random + evolutionary µ+λ).
- `evaluation.py` — `ArchitectureScore` / `ArchitectureEvaluator` (short training
  head-start, cost/latency-penalised composite quality score).
- `routing.py` — `DynamicLayerRouter` (per-token Gumbel-Sigmoid layer gates).
- `adaptive_heads.py` — `AdaptiveAttentionHeadSelector` (per-token head gating).
- `pruning.py` — `LayerPruningController` (gradient-free importance, reversible).
- `rollback.py` — `AutomaticRollback` (versioned snapshots + metric rollback).
- `self_designing.py` — `SelfDesigningTransformer` orchestrator exposing the
  backbone's duck-typed `layer_gate` / `head_weights` interface; the legacy
  module `src/cadgenesis/transformer/self_designing/self_designing.py` lives in
  this package while keeping `from cadgenesis.transformer.self_designing import
  SelfDesigningTransformer` import-compatible.
- `src/cadgenesis/transformer/moe.py` — `SparseMoEFFN` (growable experts, top-k routing,
  load-balancing auxiliary loss).  `CADTransformerBlock` now supports MoE,
  `layer_gate` (exact-skip routing) and `head_weights` modulation.

### 6.2 Layer-Integrated Memory Pools (`src/cadgenesis/memory/memory_pools.py`)
- Rewritten pool set: working, session, project, user, cad, engineering,
  manufacturing, simulation (288 slots total, preserving the existing
  `(B, 288, d)` bank-shape test contract).
- `from_config`, `retrieve` (cross-pool RAG), `refine` (differentiable per-layer
  write-back into the working pool); every encoder/decoder block reads the
  memory bank and refines it per layer.

### 6.3 Autonomous CAD Tokenizer (`src/cadgenesis/tokenizer/`)
- `vocabulary.py` — dynamic growth ops: `register` (optional pinned id),
  `remove_token`, `trim_unused`, `merge_tokens`, `split_token`,
  `remaining_slots`, `slot_capacities`.
- `evolution.py` — `TokenFrequencyTracker`, `VocabularyEvolution`
  (`analyze` / `apply` / `evolve` / `remap_sequence`), `guess_family` heuristic.
- `toon_backend.py` — `ToonBackend`: serializes CAD sequences and whole
  vocabularies (incl. slot layouts) to TOON text.  TOON itself
  (`sdk/toon.py` / `sdk/toon_extended.py`) is kept untouched as the serialization
  backend; the tokenizer does native tokenization.
- `cad_tokenizer.py` — `evolve()`, `remap_sequence()`, `toon_backend`,
  `serialize_to_toon` / `deserialize_from_toon`, and `encode_cad_token(..., auto_register=True)`.

### 6.4 Inference Engine (`src/cadgenesis/inference/engine.py`)
- `CADInferenceEngine` with greedy and beam decoding, confidence scoring from
  the model's confidence head, batch generation, TOON-serialized results and
  self-design telemetry.  Works with both `GeometryAwareTransformer` and
  `SelfDesigningTransformer`.

### 6.5 Tests
- `tests/transformer/test_self_designing.py` (28), `tests/tokenizer/test_evolution.py`
  (19), `tests/tokenizer/test_toon_backend.py` (8), `tests/test_generate.py`
  (12).  Full suite: `python -m pytest -q` → 298 passed (was 218 at time of writing).

### 6.6 Geometry Transformer Upgrade (efficient attention, geometry encodings)
Adds the remaining geometry-transformer capabilities while keeping every
default byte-identical to the pre-upgrade behaviour:

- **Geometry positional encoding** — `GeometryPositionalEncoding`
  (`src/cadgenesis/transformer/positional.py`): learned additive encoding of 3D X/Y/Z
  coordinates with optional Fourier frequency features.  Enabled via
  `ModelConfig.geometry_pos_encoding`; activated by passing `geometry_coords`
  to `encode`/`decode`/`forward`.
- **Efficient attention** — `src/cadgenesis/transformer/efficient_attention.py`:
  `SDPASelfAttention` (torch `scaled_dot_product_attention`; fused flash /
  mem-efficient kernels on CUDA), `LinearAttention` (Performer-style random
  features, O(seq_len)), and `build_self_attention` factory for the
  `ModelConfig.attention_backend` values `"math" | "sdpa" | "flash" | "linear"`.
- **Feature interaction layers** — `FeatureInteractionLayer`
  (`src/cadgenesis/transformer/interaction.py`): gated, type-biased cross-feature
  interaction sub-layer inside `CADTransformerBlock`
  (`ModelConfig.feature_interaction` / `interaction_heads`).
- **Fix** — `ConstraintAttention.constraint_bias_proj` was a dead parameter;
  it is now used as a learned per-query constraint bias when no explicit
  `constraint_mask` is supplied.
- **Tests / benchmarks** — `tests/transformer/test_geometry_upgrade.py` (32) and
  `benchmarks/attention_benchmarks.py` (``python benchmarks/attention_benchmarks.py``).
  Full suite: `python -m pytest -q` → 298 passed.

### 6.7 CAD Tokenizer Completion (versioning, statistics, validation, compression)
Completes the ten target capabilities of the Autonomous CAD Tokenizer while
keeping every pre-existing behaviour unchanged:

- **Versioning** — `src/cadgenesis/tokenizer/versioning.py`: `DEFAULT_VOCAB_VERSION`
  + `VOCAB_SCHEMA_VERSION` constants, `compare_versions`, `MigrationResult`
  (preserved/remapped/dropped counts), `migrate_vocabulary` and `remap_ids`.
  `CADVocabulary` now carries a `version` attribute; `register()` records the
  composite `parts` of merged tokens; `save()`/`load()` persist `vocab_version`,
  `schema_version` and `parts` (backward compatible with older files).
  `CADVocabulary.migrate_layout()` rebuilds a vocabulary under new slot
  capacities preserving as many ids as possible (see `last_migration`).
- **Statistics** — `src/cadgenesis/tokenizer/statistics.py`: `CorpusStatistics` and
  `compute_statistics` (per-family counts + relative shares, sequence-length
  summary, unique tokens, unknown rate, compression ratio).  Consumes str-id
  sequences, token-string sequences or `CADTokenSequence` objects.
- **Unknown-token handling** — `AutonomousCADTokenizer.is_unknown_token`,
  `validate_token` (registration + numeric decodability), `register_new_token`
  (family guessing), `unknown_rate`.
- **Compression** — `compress_sequence` (lossless greedy composite merging) /
  `expand_sequence` (recursive expansion of `TokenRecord.parts`).
- **Migration integration** — `migrate_vocabulary` / `remap_ids_to_vocab` map
  legacy id sequences into a migrated vocabulary's id space.
- **New tests** — `tests/tokenizer/test_versioning.py` (13) and
  `tests/tokenizer/test_statistics.py` (24).  Full suite: `python -m pytest -q`
  → 298 passed (was 261).
- **Benchmark** — `benchmarks/tokenizer_benchmarks.py`
  (``python benchmarks/tokenizer_benchmarks.py``): default vocab build ≈7.3 ms,
  encode 64 tokens ≈0.04 ms, decode 512 ≈0.17 ms, compression ratio ≈0.33 on a
  synthetic corpus with a registered composite token (lossless).

