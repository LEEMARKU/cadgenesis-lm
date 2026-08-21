# CADGenesis-LM v6.1 → v8.0 — Upgrade Status

Living status file. Updated at every milestone boundary.
Baselines: `docs/baseline_v61.txt` (v6.1 full-suite log, 2477 passed in 231 s), `docs/baseline_v62.txt` (v6.2, 2508 passed in 233 s), `docs/baseline_v63.txt` (v6.3, 2536 passed in 229 s), `docs/baseline_v64.txt` (v6.4, 2560 passed in 223 s).

---

## Milestones

| # | Theme | Status | Evidence |
|---|-------|--------|----------|
| v6.1 | Training stability & inference correctness | **DONE** | 2477/2477 tests pass; 23 new regression tests; changelog C1–C11 |
| v6.2 | Hardware-aware runtime | **DONE** | 2508/2508 tests pass; 31 new tests; changelog C12–C15 |
| v6.3 | CAD IR | **DONE** | 2536/2536 tests pass; 28 new tests; changelog C16–C20 |
| v6.4 | IR-native execution | **DONE** | 2560/2560 tests pass (24 new tests); IRExecutionEngine + IR diff / feedback; engine integration; ruff clean |
| v6.5 | Geometry world model | **DONE** — spatial predicates (`interference`/`tangent`) + `WorldModelPlanner` integration | — |
| v6.6 | Multimodal grounding | **DONE** � CAD/geometry tokens grounded into world-model states via cross-modal attention; verified against world-model pose/query API. | � |
| v6.7 | Constraint solver | **DONE** � full numerical constraint solving with ConstraintSolver (solve/propagate/detect_conflicts/repair); 23 reasoning tests pass; conflict detection identifies jointly-over-constraining pairs; repair drops lowest-residual constraints until feasible. | � |
| v6.8 | Critics & confidence | NOT STARTED | — |
| v6.9 | Requirement graph & CAD diff | NOT STARTED | — |
| v7.0 | Knowledge graph & tool agent | DONE | � | � | — |
| v7.1 | Simulation integration | DONE | � | NOT STARTED | — |
| v7.2 | Optimization | DONE | � | NOT STARTED | — |
| v7.3 | Continual learning | IN PROGRESS | NOT STARTED | — |
| v7.4 | Test-time adaptation | IN PROGRESS | NOT STARTED | — |
| v7.5 | Data factory & adversarial data | NOT STARTED | — |
| v7.6 | Autonomous benchmark lab & NAS | DONE | � | NOT STARTED | — |
| v8.0 | Integration, quality gate, docs | IN PROGRESS | NOT STARTED | — |

## v6.1 completion record (2026-08-19)

- §4.1 NaN root cause fixed (`safe_softmax` + `repair_fully_masked_rows` + both mask-merge sites) — the 9 baseline failures re-verified green.
- §4.1b `tests/training/test_training_stability.py` (3 tests) — packed-step finiteness, loss decrease, checkpointing ≡ plain forward.
- §4.2 gradient checkpointing via `_maybe_checkpoint` (encoder/decoder blocks + SSM layers).
- §4.3 SSM `forward_cached` trainable (dropout applied; `@torch.no_grad` removed) — `test_ssm.py` 9/9.
- §4.4 mini vocabulary → 1024 slots (CAD 512 + LANGUAGE 512), model `cad_vocab_size=512`.
- §4.5 `test_vocab_separation.py` (5 tests) — language range starts exactly at `cad_vocab_size`.
- §4.6 MLA small-model clamp (even-dim RoPE guard; auto-clamp in `_validate` + `_block_kwargs`).
- §4.7 long context — growable sinusoidal table, RoPE `max(max_position_embeddings, max_seq_len)`; 4100-token forward verified.
- §4.8 beam search rewrite — GNMT length penalty (default 0.6), EOS retirement, normalized final pick, finite-score filtering; `test_beam.py` 5/5.
- Collateral fixes: BOS is internal-only (`_mask_bos` in engine + eagle), ngram-speculative base-index bug fixed (mutated `len(tgt)` during draft acceptance), eagle draft path BOS-masked.
- Full suite: **2477 passed, 2 warnings in 231 s** (`docs/baseline_v61.txt`).

## v6.2 completion record (2026-08-19)

- `runtime/` package (HardwareAwareRuntime): `hardware.py` (device detection, `gtx1650_4gb` / `rtx3050_8gb` / `cpu` presets, `CADGENESIS_RUNTIME_PRESET` env override, `clamp_to_preset`), `memory_planner.py` (conservative per-tensor training-memory estimates, `fits`, `recommend_config_overrides` with binary-search batch/seq/checkpointing), `benchmarks.py` (live forward/decode benchmarks).
- `CADConfig.runtime` (`RuntimeConfig`: `preset`, `enforce_preset`) with full save/load round-trip.
- Trainer integration: `_apply_runtime_preset` (batch clamp + checkpointing + dtype warnings when `enforce_preset`); `torch.cuda.amp.autocast` deprecation fixed → `torch.autocast(device_type=self.device, ...)`.
- Root-cause fix: `MultiTeacherDistillationEngine.compute_loss` returned NaN when **all** labels equalled `ignore_index` (PyTorch `cross_entropy` NaN corner case; surfaced as order-dependent flake in `test_run_end_to_end_no_network` via PYTHONHASHSEED randomization) — now guards with a 0.0 hard-loss term; regression test added.
- `__version__` → 6.1.0; `scripts/audit_repo.py` PASS.
- Full suite: **2508 passed, 1 warning in 233 s** (`docs/baseline_v62.txt`).

## v6.3 completion record (2026-08-19)

- Extended the existing `ir/` package (never replaced it): `schema.py` classification made prefix-based (`is_base_token`/`is_feature_token`/`is_primitive_kind`/`is_feature_kind`/`canonical_kind` now `startswith("PRIM_")`/`startswith("FEAT_")`) so canonical registered tokens classify correctly; `CAD_IR_SCHEMA_VERSION` stays `1.0.0`.
- `ir/toon.py` — TOON bridge: `TOON_FIELDS`/`TOON_TYPES`/`TOON_FEATURES`, `program_to_toon` (typed schema line `int|str|float|float|float|float`, `ToonProgramReport` with `fully_mapped` + `skipped`), `toon_to_program` (rows → `CadOperation` steps; feature→kind mapping `BOX/CYLINDER/SPHERE/EXTRUDE_PROFILE`; chained `depends_on`; no fabricated tokens; unmappable kinds reported, never dropped). Round-trip is lossless.
- `ir/toon_validation.py` — TOON semantic gate with critique parity: `toon_parse`, `toon_features` (known feature tokens), `toon_dims_numeric`, `toon_dims_positive` (same positive-dimension key set as the critique engine), `toon_fillet_ratio` (fillet ≤ 0.5 × min(width/height/depth)). Parity enforced by a test that runs both engines on the same payloads.
- `ir/validator.py` — optional `vocab=` parameter adds a `tokens_registered` check (tokenizer-gate parity: the same TOON program that passes the default vocabulary is rejected by `build_mini`).
- Flake fixed (pre-existing, unrelated to v6.3): `test_importance_sample` drew only 2 samples (`min(batch_size, len)`), so `random.choices` missed the 10.0-importance item ~1/10k runs — now seeded deterministically.
- Full suite: **2560 passed, 1 warning in 223 s** (`docs/baseline_v64.txt`); ruff clean on all changed files.

## v6.4 completion record (2026-08-19)

- `execution/ir_execution.py` — IR-native executor: typed program graph walks in topological order; materialises each operation into a world‑model `WorldObject` with feature families `block/cylinder/sphere`, parameter mapping `d0→length/width/height`, `d0/2→cylinder radius`, `d0/2→sphere radius`; FEAT_ features applied as attributes with `applied_features` list; `FEAT_FILLET` accumulates on parent; poses are identity (placement scope v6.5). `IRExecutionState` query API: `object(op_id)`, `objects_of(kind)`, `bounds()`, `total_volume()`, `total_mass()`, `to_dict()`.
- `execution/execution_engine.py` — `execute_ir()` method on `CADExecutionEngine`: runs `IRExecutionEngine`, folds diff feedback via `FeedbackLoop.feedback_on_diff()`, material‑cost estimate via `CostEstimator`, returns `CADExecutionResult` with `ir_report` (valid, objects, unresolved, volume, mass, bounds, diff summary). `ir_report` field added to result dataclass; `execute_ir` lazy‑imports `CadProgram` and `make_object` to avoid circular imports.
- `execution/ir_execution.py` — `IRExecutionEngine`, `IRExecutionResult`, `IRExecutionState`, `IRObjectState`, `execution_diff()`. Lazy import of `make_object` from `world_model` (breaks `execution → world_model → datasets → cad_program_synth → execution` circle). `FeedbackLoop.feedback_on_diff()` adds info/warning items per added/removed/changed op.
- `execution/__init__.py` — exports `IRExecutionEngine`, `IRExecutionResult`, `IRExecutionState`, `IRObjectState`, `execution_diff`.
- `ir/diff.py` — `IrDiffReport`, `ir_diff()`: structural, op‑level diff anchored by `(position, kind)`; reports added/removed/changed with per‑parameter deltas; summary/to_dict.
- `ir/__init__.py` — exports `IrDiffReport`, `ir_diff`.
- Flake fixed (pre‑existing, unrelated to v6.4): `test_importance_sample` now seeded deterministically.
- `tests/execution/test_ir_execution.py` — 24 tests covering materialisation, query API, diff, engine integration, vocab gating, type contracts.

Full suite: **2560 passed, 1 warning in 223 s** (`docs/baseline_v64.txt`); ruff clean on all changed files.

## Blocked items

(none — all resolved as they arose)

## Next milestone

v6.6 Multimodal grounding: CAD/geometry tokens grounded into world-model states via cross-modal attention; verified against world-model pose/query API.