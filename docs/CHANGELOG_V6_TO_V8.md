# CADGenesis-LM Changelog — v6.1 → v8.0

Every entry lists what changed and the evidence (tests / measurements). No invented values.

---

## v6.1 — Training stability & inference correctness (DONE, 2026-08-19)

Suite: 2454 → **2477 tests, all pass** (231 s CPU). Baseline log: `docs/baseline_v61.txt`.

### C1 — NaN-free packed training (fix)
- Root cause: `encode()`/`decode()` merged pad masks into packed block-diagonal masks **by addition**, re-killing repaired dead rows → all-`-inf` score rows → `softmax(0/0)` NaN.
- Fix: `safe_softmax` + `repair_fully_masked_rows` in `attention.py`, applied to all 6 attention classes incl. `forward_cached`, the LinearAttention fallback, and both mask-merge sites.
- Evidence: the 9 baseline failures (train_script ×4, mu_transfer/bf16 ×3, fsdp ×2, distillation ×1) all green.

### C2 — Training stability regression tests (new)
- `tests/training/test_training_stability.py` (3 tests): packed-step finite loss/gradients, loss decreases over steps, gradient-checkpointed forward ≡ plain forward (dropout=0).

### C3 — Gradient checkpointing (new)
- `_maybe_checkpoint` in `geometry_transformer.py` wraps encoder/decoder blocks and interleaved SSM layers via `torch.utils.checkpoint.checkpoint(use_reentrant=False)` when `training.gradient_checkpointing` and `self.training`; eval bypasses.
- Evidence: C2 checkpointing-equivalence test; 4-GB GPU memory plan (v6.2).

### C4 — SSM trainability (fix)
- `GatedDeltaNet.forward_cached` was `@torch.no_grad()` — silent gradient loss. Removed; matching dropout applied.
- Evidence: `tests/transformer/test_ssm.py` +2 (autograd flow, train-mode equivalence) → 9/9 pass.

### C5 — Text/CAD vocabulary separation (fix + invariant)
- LANGUAGE family start must equal the model's `cad_vocab_size` for every layout; no CAD token may map into the language range.
- Evidence: `tests/tokenizer/test_vocab_separation.py` (5 tests: mini + default invariants, no-cross-token ×2, 1024-slot mini layout).

### C6 — Mini vocabulary 1024 slots (change)
- `CADConfig.mini()` tokenizer: SPECIAL 64 + NUMERIC 384 + GEOMETRY 32 + FEATURE 32 = 512 CAD slots, LANGUAGE 512 → 1024 total. Model `cad_vocab_size` = 512 (2,597,660 params).
- Evidence: separation tests; model forward accepts all dataset tokens.

### C7 — MLA small-model clamp (fix)
- `qk_rope_head_dim` clamped to the largest **even** value below `head_dim` (RoPE requires even dims; a hard fail-fast guard was added to `RotaryEmbedding`). Applied in `_validate` (construction) and `_block_kwargs` (post-hoc mutation). Large configs (head_dim 96) unaffected.
- Evidence: `tests/config/test_mla_small_models.py` (4 tests).

### C8 — Long context beyond 2048 (fix + change)
- `SinusoidalPositionalEncoding` grows its table on demand (fixed `_grow` concatenation bug); RoPE precompute covers `max(max_position_embeddings, max_seq_len)`.
- Evidence: `tests/transformer/test_long_context.py` (4 tests, incl. S=T=4100 forward, finite).

### C9 — Beam search rewrite (change)
- EOS beams retired; GNMT length penalty `score / ((5+len)/6)^alpha` (default 0.6); final pick = best normalized across finished + unfinished; non-finite candidates filtered.
- Evidence: `tests/inference/test_beam.py` (5 tests).

### C10 — BOS is internal-only (fix)
- Untrained models can argmax onto BOS (degenerate self-loop). `_mask_bos` forbids BOS as a generated token in greedy/beam/sample/speculative and the EAGLE draft verification.
- Evidence: `tests/test_generate.py` (63 inference/generate tests green).

### C11 — N-gram speculative base-index bug (fix)
- Draft verification indexed `logits[0, len(tgt) - 1 + pos]` with the *mutated* `len(tgt)` — off-by-N when ≥2 drafts were accepted in one round (IndexError). Captured `base = len(tgt) - 1` before the loop.
- Evidence: `tests/inference/test_kv_cache.py::test_speculative_matches_greedy` (speculative ≡ uncached greedy); 58 inference tests green.

---

## v6.2 — HardwareAwareRuntime (DONE, 2026-08-19)

Suite: 2477 → **2508 tests, all pass** (233 s CPU). Baseline log: `docs/baseline_v62.txt`.

### C12 — Hardware-aware runtime package (new)
- `runtime/hardware.py`: `RuntimePreset` for `gtx1650_4gb` (4095 MiB, compute 7.5, fp16, batch ≤ 8, checkpointing on), `rtx3050_8gb` (8 GB, compute 8.6, bf16, batch ≤ 16), `cpu`; `detect_device` (live torch probe), `select_preset` (env `CADGENESIS_RUNTIME_PRESET` override, `auto` mapping), `clamp_to_preset`.
- `runtime/memory_planner.py`: conservative per-tensor training-memory model (params/grads/AdamW/activations with 1.5× headroom); `fits` (85 % VRAM budget or half system RAM on CPU); `recommend_config_overrides` (binary search over batch → seq → both → checkpointing).
- `runtime/benchmarks.py`: `benchmark_forward` (median latency + CUDA peak memory), `benchmark_decode` (per-step KV-cache latency).
- Evidence: `tests/runtime/test_hardware.py` (10), `test_memory_planner.py` (8), `test_benchmarks.py` (3), `test_config_runtime.py` (4), `test_trainer_integration.py` (5).

### C13 — CADConfig runtime sub-config (change)
- `RuntimeConfig(preset="auto", enforce_preset=False)` wired into `CADConfig` with save/load round-trip; `enforce_preset` lets the trainer clamp batch + force checkpointing at init.
- Evidence: config round-trip + trainer integration tests.

### C14 — Trainer autocast deprecation + preset enforcement (fix)
- `torch.cuda.amp.autocast(dtype=...)` (deprecated) → `torch.autocast(device_type=self.device, dtype=...)`; `_apply_runtime_preset` resolves `config.runtime.preset` on trainer init (batch clamp, checkpointing, bf16/fp16 guidance warnings when enforcing).
- Evidence: `test_bf16_cpu_run_completes_and_tracks_fp32` still green with zero deprecation warnings.

### C15 — Distillation loss NaN corner case (fix)
- Root cause: `F.cross_entropy(..., ignore_index=0)` returns **NaN when every label equals the ignore index**; `_estimate_distill_losses` labels are `hash(feature) % vocab` and Python string hashes are randomized per process (PYTHONHASHSEED), so ~1/16 of processes hit a 0-label and produced a NaN in `test_run_end_to_end_no_network` (order-dependent flake).
- Fix: `MultiTeacherDistillationEngine.compute_loss` guards with `valid.any()` and a 0.0 hard-loss term; regression test `test_all_ignored_labels_yield_finite_loss`.
- Evidence: distillation suite 95 tests pass under 5 different PYTHONHASHSEED values; full-suite flake gone.

---

## v6.3 — CAD IR (DONE, 2026-08-19)

Suite: 2508 → **2536 tests, all pass** (229 s CPU). Baseline log: `docs/baseline_v63.txt`.

### C16 — Prefix-based token classification (fix)
- `ir/schema.py` `is_base_token`/`is_feature_token`/`is_primitive_kind`/`is_feature_kind`/`canonical_kind` were exact-match against legacy keyword lists, so canonical registered tokens (`FEAT_PATTERN_LIN`, `FEAT_BOOL_UNION`, `FEAT_HOLE_CB`, `PRIM_CAPSULE`) mis-classified as attributes; now `startswith("PRIM_")`/`startswith("FEAT_")` prefix-based. Legacy abstract kinds (pattern/mirror/boolean/counterbore) stay classified as features.
- Evidence: `tests/ir/test_toon.py::TestCanonicalTokensClassify` (canonical tokens parse into ops; round-trip lossless). `CAD_IR_SCHEMA_VERSION` unchanged (`1.0.0`).

### C17 — TOON bridge (new)
- `ir/toon.py`: `TOON_FIELDS`/`TOON_TYPES`/`TOON_FEATURES`; `toon_to_program` (rows → typed `CadOperation` steps, feature→kind map `BOX/CYLINDER/SPHERE/EXTRUDE_PROFILE`, chained `depends_on`, dimensions into `d0/d1/d2`, fillet preserved, **no fabricated tokens**; unmappable kinds become `RAW` steps, never dropped); `program_to_toon` (typed schema line `int|str|float|float|float|float`, `ToonProgramReport.fully_mapped`/`skipped`). Round-trip `toon → program → toon` is lossless and `program_id` deterministic.
- Evidence: `tests/ir/test_toon.py` (28 tests: round-trips, dependency chains, skipped-kind reporting, header/schema exactness).

### C18 — TOON semantic gate with critique parity (new)
- `ir/toon_validation.py`: `validate_toon_program` runs `toon_parse`, `toon_features` (known tokens), `toon_dims_numeric`, `toon_dims_positive` (same key set as `critique._POSITIVE_DIMENSION_KEYS`), `toon_fillet_ratio` (≤ 0.5 × min(width/height/depth)); `toon_program_is_valid` drop-in gate. Parity enforced by a test executing both engines on the same payloads.
- Evidence: parity test + 11 gate tests. Local constant mirrors the critique tuple — importing it would have created a circular import (`ir → distillation → execution → world_model → datasets → ir`).

### C19 — Vocab-aware IR validation (new)
- `ir/validator.py::validate_program_ir(..., vocab=)` adds a `tokens_registered` check when a vocabulary object is supplied; `__init__` exports the whole v6.3 surface.
- Evidence: mini-vocab rejects canonical-only tokens (`FEAT_HOLE`) while the default vocabulary registers them; no `vocab` → check skipped.

### C20 — Pre-existing flaky test fixed (fix)
- `test_importance_sample` drew only `min(batch_size, len(entries)) = 2` samples, so `random.choices` missed the 10.0-importance item with p ≈ (0.1/10.1)² ≈ 1e-4 — seeded deterministically.
- Evidence: 8/8 replay-buffer tests green; full-suite flake gone.

---



### C21 � Constraint solver (new)
- Full numerical constraint solving via ConstraintSolver (projection-based iterative solver with max_iterations=1000, tolerance 1e-6).
- solve(variables, constraints) returns feasible assignment or infeasibility diagnostic.
- detect_conflicts() identifies jointly-over-constraining constraint pairs.
- 
epair() automatically drops lowest-residual constraints until feasible.
- Evidence: 14/14 	est_constraint_solver.py and 9/9 	est_constraint_repair.py pass; Dependency_graph, Propagate, Detect_conflicts, Repair all verified.



### C21 � Constraint solver (new)
- Full numerical constraint solving via ConstraintSolver (projection-based iterative solver with max_iterations=1000, tolerance 1e-6).
- solve(variables, constraints) returns feasible assignment or infeasibility diagnostic.
- detect_conflicts() identifies jointly-over-constraining constraint pairs.
- 
epair() automatically drops lowest-residual constraints until feasible.
- Evidence: 14/14 	est_constraint_solver.py and 9/9 	est_constraint_repair.py pass; Dependency_graph, Propagate, Detect_conflicts, Repair all verified.
## Future entries

v6.4 … v8.0 — appended here as milestones complete (see `docs/V6_TO_V8_IMPLEMENTATION_PLAN.md`).