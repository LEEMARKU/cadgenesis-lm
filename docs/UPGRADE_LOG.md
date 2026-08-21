# Upgrade Log

Mission: 10/10 Autonomous Upgrade Engineering Mission (brief sections 1-39).
Rules: inspect before modify; smallest safe change; never delete/weaken functionality
or tests; measure before claiming; `NOT MEASURED` for unmeasurable values; no gaming.

## M1 - Transformer & Tokenizer (COMPLETE)

### Changes
- `src/cadgenesis/tokenizer/cad_tokenizer.py`
  - Added `_LEGACY_FEATURE_TOKENS` (12) and `_LEGACY_GEOMETRY_TOKENS` (21) covering
    every dataset-layer token name (EXTRUDE, BOX, SKETCH_RECT, ...).
  - `_register_legacy_numeric_tokens`: raw-mm tokens `NUM_0..NUM_255` (unpadded,
    digits len < 3 decode as raw mm, e.g. `NUM_80` = 80.0 mm).
  - `_register_legacy_cad_tokens`: idempotent legacy name registration; wired into
    `build()` and `build_mini()`.
  - `build_mini()` capacity layout tightened so every CAD-family id < 512
    (mini model sizes lang_vocab_size=512): SPECIAL 64 / NUMERIC 384 / GEOMETRY 32 /
    FEATURE 32 / LANGUAGE 512.  Registers raw NUM_0..255 + padded NUM_000..NUM_099
    (dataset values <= 155 mm quantize to bins <= 39) + legacy names.
  - `decode_length`: unpadded (len < 3 digits) = raw mm; padded = quantizer bin.
  - Public `unk_id` property; `encode_text` normalizes engineering notation first.
- `src/cadgenesis/tokenizer/engineering.py` (NEW)
  - `normalize_engineering_notation`: deterministic, idempotent rewrite of
    engineering notation to plain text (diameter O25, radius R12.5, thread M8x1.25,
    tolerance +/-0.02, exponent 1e-5, negative numbers, coordinate/vector tuples,
    unit expansion).  Plain text passes through unchanged.
  - `parse_engineering_tokens`: split prompt into word tokens after normalization.
- `src/cadgenesis/datasets/cad_program_synth.py`
  - `_num_token(value)`: raw `NUM_<v>` for v < 100, canonical `encode_length` bin
    for v >= 100 (unpadded names >= 100 would collide with padded bin tokens).
- Tests
  - `tests/test_cad_tokenizer.py` (27): keyword/numeric/unit preservation,
    engineering notation, coordinates, idempotency, round-trip, unknown rate,
    token efficiency, encode_cad_sequence round-trip.
  - `tests/tokenizer/test_legacy_vocab_coverage.py`: dataset tokens covered by full
    + mini vocab, family checks, canonical preserved, numeric round-trip, mini ids
    < lang_vocab_size (regression), model forward accepts dataset tokens.
  - `tests/inference/test_model_boundaries.py` (5): vocab boundary, block_size
    boundary, finite loss, NaN logits, gradient step.
  - Updated `tests/evaluation/test_tokenizer_metrics.py` (NUM_0..NUM_19 coverage
    40.0/total; mini sum 49.0/total) and
    `tests/datasets/test_cad_program_synth.py` (mini vocab now registers every
    dataset token: `missing == []`).

### Evidence (measured)
| Metric | Before (audit) | After (M1) |
|---|---|---|
| Full suite passed | 2242 | 2285 |
| Full suite failed | 22 | 20 |
| EAGLE / rlvr suites | pass | pass (289 in tokenizer/eagle/rlvr batch) |
| ruff check src tests | clean | clean |
| ruff format --check | clean | clean |
| mypy src | 452 files | 452 files, no issues |

### Remaining failures (20, all pre-existing)
- adapters 6 (peft lora delta, promotion x5) - M4
- distillation 10 (consensus x2, critique, hard_labels x3, pipeline x3, soft_labels,
  synthetic) - M4
- continual_learning 1 (adapter isolation) - M4
- evaluation 1 (`test_dimension_relative_error`) - M9
- training 1 (resume replay) - M4
- rlvr 1 (`test_eagle_train_and_speculative`) - M4

### Root cause recorded
Mini vocab ids exceeded lang_embed range mid-change (ids up to 714 > 512) because
NUMERIC capacity was raised without accounting for the fixed family offset layout;
fixed by tightening capacities (max id 490 < 512) + regression test
`test_mini_ids_stay_within_lang_embed_range`.

## M2 - CAD-IR (COMPLETE)

### Changes
- `src/cadgenesis/ir/` (NEW package)
  - `schema.py`: token-role classification (base primitives `BOX`/`CYLINDER`/
    `SPHERE`/`SKETCH_RECT`/`SKETCH` + `PRIM_*`; features + `FEAT_*`; numeric
    prefixes `NUM_`/`ANG_`/`RAT_`), canonical kind mapping, param decode
    (unpadded `NUM_<v>` = raw mm; padded = quantizer bin), `CAD_IR_SCHEMA_VERSION
    = "1.0.0"` and `is_schema_compatible` (major must match, minor/patch must
    not exceed consumer).
  - `program.py`: `CadOperation` (op_id = deterministic content hash, kind,
    params in mm, depends_on, exact token slice, position) and `CadProgram`
    (program_id content hash, schema_version, steps, `topological_order`,
    `is_cyclic`, `to_tokens` lossless round-trip, to_dict/from_dict,
    to_json/from_json).
  - `parser.py`: `parse_program` — total parser (never fails): base keyword
    opens PRIMITIVE op, feature keyword opens FEATURE op, numeric attaches as
    decoded param `d0..dN`, other tokens become attributes; linear dependency
    chain; every token assigned in order -> `to_tokens() == tokens` always.
  - `validator.py`: `validate_cad_program`/`validate_program_ir` — schema
    version, non-empty steps, unique op ids, deps resolve, acyclic, params in
    range [0,1000] mm, base present (mirrors legacy gate exactly — `EXTRUDE`
    counts as base), round-trip identity.  `CadProgramReport` with checks +
    summary.
- Integrations (all additive)
  - `datasets/cad_program_synth.py`: `_validate_tokens` now runs the legacy
    gate AND the IR gate.  Verified composition-neutral: 0/300 dataset
    records rejected by the IR gate (seed 42); dataset tests unchanged.
  - `execution/freecad_engine.py`, `opencascade_engine.py` (via
    `_primitives_from_program`): accept `CadProgram` objects (duck-typed
    `.to_tokens()`).
  - `execution/execution_engine.py`: `execute_and_evaluate` accepts
    `CadProgram`; records `program_id` in `parametric_json`.
- Tests: `tests/ir/test_cad_ir.py` (57 tests): classification, lossless
  round-trip on 7 representative shapes + unknown tokens + empty, dependency
  chain + topological order, decoded params, attributes, validation failures
  (no base, unknown deps, duplicate ids, out-of-range params, round-trip
  violation), JSON/JSONL serialization stability, deterministic content IDs,
  schema versioning rules, dataset integration (every record passes IR gate,
  ID uniqueness per content, legacy-gate parity, engine integration x4).

### Evidence (measured)
| Metric | After M1 | After M2 |
|---|---|---|
| Full suite passed | 2285 | 2342 |
| Full suite failed | 20 | 20 (unchanged, all pre-existing) |
| ruff check / format | clean | clean |
| mypy src | 452 files | 457 files, no issues |

## M3 - Dataset scale-up (COMPLETE)

### Changes
- `src/cadgenesis/datasets/curriculum.py` (NEW): multi-category curriculum
  dataset pipeline.
  - 9 record categories: nl2program, nl2ir, program2explanation,
    geometry2description, error2correction, constraint, parameter, tool,
    planning — record shape `{"text", "cad", "type", "program_id", "score",
    "quality"}` (superset of legacy `{"text","cad"}`, loads with
    `CADJsonlDataset` unchanged).
  - `_sample_program_fine`: reuses the validated template set with 1 mm
    dimension granularity (5..155 mm, 151 options/slot vs legacy 5 mm/31)
    — every value stays inside registered raw-mm / bin token conventions.
  - `quality_filter`: syntax -> schema (CAD-IR) -> execute -> geometry ->
    constraint -> dedup -> weighted score (threshold 0.85).  Dedup is exact
    (content-hash program_id): dimension variants are deliberately kept
    (they are the numeric curriculum); MinHash near-dup remains available
    at load time.
  - `adversarial_records`: 5 perturbation kinds (drop base incl. legacy
    EXTRUDE, all-attributes, empty, unknown-tokens, numeric-only) — all
    guaranteed invalid by construction.
  - `make_splits`: deterministic, type-stratified, leakage-free
    (program_id disjoint across splits).
  - `write_curriculum_jsonl`: train/val/test JSONL + `dataset_manifest.json`
    (counts, sha256 content digests per split, per-type counts) — the
    reproducibility contract.
  - `_truncate_balanced`: per-category round-robin truncation so every
    category reaches the target even when dedup shrinks the pool.
- Tests: `tests/datasets/test_curriculum.py` (14 tests): per-seed
  determinism, all 9 categories present + balanced, quality scores >= 0.85,
  adversarial set 100% rejected / valid set 100% kept, program_id
  uniqueness, leakage-free stratified splits, invalid fractions raise,
  JSONL + manifest counts/digests, digest stability per seed, existing
  pipeline loads curriculum files, mini-vocab covers all curriculum tokens.

### Evidence (measured)
| Metric | After M2 | After M3 |
|---|---|---|
| Dataset records | 500 (1 category, no splits) | 10,000 (9 categories, 8,002/999/999 splits) |
| Dataset tokens | 51 unique | measured in data/curriculum |
| Full suite passed | 2342 | 2356 |
| Full suite failed | 20 | 20 (unchanged, all pre-existing) |
| ruff / format / mypy | clean | clean (458 files) |
| Curriculum generation | n/a | 10,000 records in ~10 s (deterministic, seed 0) |
| Adversarial rejection | n/a | 100% (quality filter precision) |
| Dedup | none | exact content-hash + per-type balance |

Files: `data/curriculum/train.jsonl` (8,002), `val.jsonl` (999),
`test.jsonl` (999), `dataset_manifest.json`.

### Next
- M4: training — reproducible mini-config SFT run on the curriculum dataset
  (CPU/4 GB), checkpointing, config hash, seed control; fix the 9
  training-family failures (adapters 6, distillation 10 -> owned by M4).

## M4 - Training runs + training-family fixes (COMPLETE)

### Changes
- Training-family test fixes (7):
  - `adapters/promotion.py`: `promote()` signature simplified to
    `(metadata, metrics, criteria=None)` (dead `adapter_id` param removed,
    grep-verified no other callers).  4 stale tests updated with
    `"samples": 1.0` — the samples gate (`test_min_samples_gate`) is the
    intended semantics; the older tests contradicted it.
  - `adapters/test_peft.py`: expected-value assertion moved to the
    wrapper level (old test compared a wrapper-delta against the *full*
    model output).
  - `continual_learning/adapter_isolation.py`: `release()` now re-applies
    the masks of still-active tasks (previously it only restored the
    snapshot); test assertion fixed to match the test's own documented
    semantics.
- Distillation fixes (12 tests across 6 files):
  - `consensus.py`: removed `round(agreement, 4)` in `toon_votes` and
    `consensus_logits` — exact-fraction assertions (2/3, 2.0/2.8, 0.5) are
    the contract; rounding was destroying them.
  - `critique.py`: an unparsable TOON payload now scores 0.0 (was
    1 - 0.25 = 0.75) — total failure semantics.
  - `hard_labels.py`: `extract()` now detects teacher positions whose
    argmax value IS the ignore_index (whole rows pre-filled with -100)
    and marks them ignored unconditionally (the old `ignore_index !=
    -100` guard was inverted logic that let ignored rows leak into the
    loss).  Two test bugs fixed: operator-precedence broken assertion
    (`not batch.confidence[1].any() > 0.8` -> `batch.confidence[1] <=
    0.8`), and the loss-ready test used a min_confidence that masked
    *every* position (CE over an empty target is nan by torch design) —
    the test now pre-ignores one row and keeps the rest.
  - `distill_pipeline.py`: `DistillationLossPipeline.compute_loss` no
    longer assumes 3-D logits (callers pass 2-D (B, V)); `generate_dataset`
    guards the pass-rate division when `attempts == 0` (run(0) is safe).
  - `soft_labels.py`: `kl_loss` uses `log_target=True` on both log-softmax
    sides — identical teacher/student logits now produce exactly 0.0.
  - `synthetic.py`: zero-noise perturbation preserves the original token
    text (previously reformatted numerics to 2 decimals, breaking
    identity).
- Training evidence run (reproducible SFT, CPU):
  - Run A: 2,000 records / 3 epochs — train 8.0159 -> 4.7022,
    val 7.5796 -> 3.9653.  Run B: 5,000 records / 6 epochs —
    train 6.8868 -> 1.9062, val 4.6820 -> **1.8868** (random baseline
    ln(412) = 6.02).  Checkpoints per epoch + best, `run_digest` in every
    checkpoint, seed 0, WSD schedule, packed collation, 2,966,306 params.
  - Trained model generates valid CAD programs: greedy `SLOT EXTRUDE
    <eos>`; **20/20 greedy generations parse to a CadProgram and pass all
    7 CAD-IR validation checks**.  The 3-epoch model's greedy argmax
    stuck on `<bos>` (BOS is the most frequent token); sampling or longer
    training escapes it — documented as undertraining, not a pipeline bug.
  - `_run_digest` now includes `max_records`: two runs with different
    `--max-records` previously shared a digest, so resume could silently
    continue a different experiment.  `max_epochs` stays normalized out
    by design (extending the horizon on resume is the same run).
- Baseline-doc correction: `docs/UPGRADE_BASELINE.md` claimed "no
  checkpoints" but `checkpoints/` contains 8 pre-existing run dirs with
  artifacts — recorded here; M4's own runs are new.

### Evidence (measured)
| Metric | After M3 | After M4 |
|---|---|---|
| Full suite passed | 2356 | 2375 |
| Full suite failed | 20 | 1 (pre-existing M9: `test_dimension_relative_error`) |
| ruff check / format | clean | clean (2 files auto-formatted) |
| mypy src | 458 files, clean | 458 files, clean |
| Distillation suite | 82/94 | 94/94 |
| Adapt + continual suites | 118/125 | 125/125 |
| Training run A (2k rec, 3 ep) | n/a | train 8.02->4.70, val 7.58->3.97 |
| Training run B (5k rec, 6 ep) | n/a | train 6.89->1.91, val 4.68->1.89 |
| Generated-program validity | n/a | 20/20 pass CAD-IR validation |
| Checkpoint reproducibility | n/a | run_digest + dataset sha256 + vocab_tokens in checkpoint; resume guarded |

Files: `checkpoints/m4-curriculum/`, `checkpoints/m4-curriculum-6ep/`.

### Next
- M5: CAD reasoning — inspect `reasoning/` planner/validator first, then
  exercise/fix; re-check the `cli` entrypoints that were dropped.

## M5 - Program-level reasoning bridge (COMPLETE)

### Changes
- `src/cadgenesis/reasoning/program_reasoning.py` (NEW): closes the loop
  between the M2 CAD-IR and the reasoning stack (Pillar 7).
  - `ProgramReasoningEngine` — reason about (and repair) a `CadProgram`:
    * **rules** — DFM rules consistent with `ManufacturingRules` thresholds
      (min wall 0.8 mm, max size 1,000 mm, min hole 1.0 mm); errors block,
      warnings inform (`hole_too_small` fires as a warning);
    * **constraints** — every decoded parameter becomes a bounded variable
      `[0.8, 1000]` with min/max constraints, solved by `ConstraintSolver`;
    * **geometry** — PRIM_* ops map to `GeometryReasoner.Primitive` with the
      execution-backend parameter order (box = length/width/height, cylinder
      = radius/height, ...); missing params fall back to toolkit defaults;
    * **manufacturing** — DFM `part` dict built from the program's min
      dimension + first HOLE/COUNTERBORE/THREAD parameter.
  - Honest geometry stage: flat token programs carry no spatial layout, so
    the built-in AABB-interference stage is replaced by a "well-formed
    dimensions" check (interference is explicitly *not* claimed).
  - `repair(program, extra_constraints=None)` — constraint relaxation via
    `ConstraintSolver.repair`; bridge bounds are always satisfiable, so
    caller-injected conflicting design rules are what gets relaxed.
  - `benchmark(programs)` — measured pass rate, per-stage failure counts,
    timing.
- Tests: `tests/reasoning/test_program_reasoning.py` (14 tests): primitive
  mapping + defaults + feature-op skipping, context keyed by program_id,
  valid programs pass all stages, wall-too-thin blocked by rules AND
  manufacturing AND geometry (three layers), oversize blocked, hole
  warning fires in rules while manufacturing catches the same violation,
  repair relaxes conflicting extra constraints (assignment adjusted),
  benchmark aggregation, thresholds consistent with `ManufacturingRules`.

### Evidence (measured)
| Metric | After M4 | After M5 |
|---|---|---|
| Full suite passed | 2375 | 2389 |
| Full suite failed | 1 | 1 (unchanged: pre-existing M9) |
| Reasoning suite | 231 | 245 |
| ruff check / format | clean | clean (705 files) |
| mypy src | 458 files | 459 files, no issues |
| Curriculum programs through reasoning | n/a | 100/100 passed (mean 0.08 ms) |
| Model-generated programs through reasoning | n/a | 50/50 passed (mean 0.09 ms) |
| Thin-wall program | n/a | rejected (rules + manufacturing + geometry) |
| Repair (design rule vs bounds) | n/a | assignment adjusted to satisfy rule, 0 dropped |

### Next
- M6: RAG — inspect `memory/` + `retrieval/`; exercise and fix; measure.

---

## M6 — RAG (Retrieval-Augmented Generation)

### Goal
Bring the M4-trained model and the memory/retrieval subsystem together: a
CAD-command `MemoryStore` index over the curriculum corpus, honest retrieval
metrics, and prompt augmentation for generation — all measured, nothing
claimed.

### Defects found and fixed (during M6 — root-cause work that invalidated the
M4 "trained" claim, see M4 addendum below)

1. **`token_coverage()` never covered the text side** — `train.py` registered
   only CAD tokens from `record["cad"]`; the encoder's `lang_embed` saw
   `<unk>` for every word, so the M4 runs could only learn an
   *unconditional* CAD-sequence prior.  Fixed in `train.py`: the language
   vocabulary is now built from the dataset texts via
   `tokenizer.build_lang_vocab(...)` (deterministic sorted word set, 292
   words for the curriculum corpus, fits the mini `lang_embed` 512 rows).
2. **Learnable `geom_scale` collapses cross-attention** — `GeometryAttention`
   scaled attention scores by a trainable per-head parameter; training drove
   it to ~0.03 in every decoder block, softmax went uniform over the keys,
   and the decoder actively ignored the encoder (measured: first-token loss
   real-src 2.5662 *worse* than blank-src 2.5446; layer trace showed encoder
   diff 0.339 collapsing to geometry-attn diff 0.0013).  Fixed in
   `src/cadgenesis/transformer/attention.py`: `geom_scale` is now a frozen
   buffer at 1.0 (checkpoint-key compatible), so cross-attention must learn
   informative scores.  After retraining: different prompts yield different
   programs and real-src first-token loss (2.3879) beats blank (2.3962).

### M4 addendum (re-trained model, genuine text conditioning)
| Run | Data / epochs | Final val loss | Conditioning evidence |
|---|---|---|---|
| M4 Run B (baseline claim) | 5,000 / 6 | 1.8868 | **None** — `<unk>`-only src, unconditional prior |
| M4 Run C (12ep) | 5,000 / 12 | 3.2131 | none (still collapsed, no decay) |
| M4 Run C (24ep) | 5,000 / 24 | 1.9573 | first-token real 2.5662 vs blank 2.5446 (ignores src) |
| **M4-xattn-frozen (24ep)** | 5,000 / 24 | **1.8676** | first-token real 2.3879 vs blank 2.3962; distinct programs per prompt |

`checkpoints/m4-curriculum-xattn-frozen/best_checkpoint.pt` is the model used
for M6 measurements (digest `5257fece...`; note: digest covers config/data/
seed — the tokenizer + attention fixes are documented here, not hashed).

### Deliverables
- New package `src/cadgenesis/rag/` — `CADRAGEngine`:
  - `index_jsonl` / `index_record` — indexes `{text, cad}` records into
    `MemoryStore(name="cad-rag")` keyed by CAD-IR `program_id` (computed via
    `parse_program` when absent);
  - `retrieve(query, k)` — `MemoryRetrieval` keyword scoring;
  - `precision_at_k` (exact-key) and `template_precision_at_k`
    (operation-kind template, e.g. `(PRIM_BOX, FEAT_EXTRUDE)`) + honest
    `benchmark_retrieval(queries, k, template=True)` aggregation;
  - `augmented_prompt(query, top_k)` — query + `reference:` exemplar lines
    (bounded by the mini tokenizer's 32-token text window);
  - `generate(engine, query, ...)` — augmented greedy/sample via the existing
    `CADInferenceEngine`.
- Tests: `tests/rag/test_rag_engine.py` (12 tests) — index/unindex, id
  derivation from CAD-IR, retrieve ranking, both precision metrics, top-k
  bounds, augmentation format, generation returns tokens/text, benchmark
  aggregation, exact-key benchmark mode.

### Evidence (measured)
| Metric | Value |
|---|---|
| Full suite passed | 2401 (was 2389 after M5) |
| Full suite failed | 1 (unchanged: pre-existing M9) |
| RAG suite | 12/12 |
| ruff check / format | clean (708 files) |
| mypy src | 461 files, no issues |
| Index time (8,002 train records) | 182 ms |
| Retrieval mean latency (100 val queries, k=5) | 122 ms |
| Template precision@5 (100 val queries) | 0.98 |
| Template precision@1 (100 val queries) | 0.84 |
| Exact-key precision@5 across splits | 0.0 (expected — leak-free splits) |
| Greedy generation A/B (50 val prompts, M4-xattn-frozen) | IR-valid 38/50 both; template-match 6/50 baseline vs 4/50 RAG; mean token overlap 0.133 → 0.141 (RAG +6%) |
| Conditioning probe | distinct outputs per prompt; real-src first-token loss beats blank |

### Next
- M7: tool calling — inspect `tools/` + `execution/`; exercise and fix;
  measure.

---

## M7 — Tool Calling

### Goal
Give agents a real, measured tool-calling surface: schemas, a registry with
permission enforcement, an executor bound to the actual CAD execution
backends, and an agent-side bridge speaking the existing
`AgentRequest` / `AgentResult` envelope.

### Finding
Tool calling was entirely unimplemented (consistent with
`docs/UPGRADE_BASELINE.md` §10): no tool registry, schemas or dispatcher
anywhere in `src/`.  The building blocks existed but were unwired:
`ExecutionAdapter` (agents/integration.py), the FreeCAD / OpenCascade
backend engines, `TopologyAnalyzer` / `CostEstimator` /
`ManufacturabilityAnalyzer`, `ExportEngine`, and the CAD-IR seam
(`CadProgram.to_tokens()`).

### Deliverables
- New package `src/cadgenesis/tools/`:
  - `schema.py` — `Permission` (read < execute < admin, with
    `permission_allows`), `ParameterSpec` (typed args: string/number/
    boolean/list/program), `ToolDefinition`, `ToolCall`, `ToolResult`;
  - `registry.py` — `ToolRegistry` (thread-safe register/unregister/
    get/list + `validate_call`: unknown tool, missing/unknown/type-wrong
    parameters, permission enforcement);
  - `executor.py` — `ToolExecutor` with six built-in tools bound to the
    real backends: `validate_program` (read, CAD-IR),
    `execute_program` (execute, FreeCAD + OpenCascade analytic backends,
    backend argument validated), `analyze_brep` (read, B-Rep topology from
    the program's box dims), `estimate_cost` (read, CostEstimator),
    `manufacturing_check` (read, DFM rules; severity strings normalized to
    a numeric `max_severity`), `export_program` (admin, exporter mesh
    triangulated from the program dims).  `dispatch` never raises — schema
    and permission failures come back as `ToolResult(ok=False)`.
  - `agent.py` — `AgentToolBridge` (action `"tool_call"`): payload
    `{tool, arguments, permission?}` → validated dispatch → `AgentResult`,
    so any role agent can emit tool calls.
- Tests: `tests/tools/test_tools.py` (30 tests) — permission order,
  schema validation, registry lifecycle + validation failures, all six
  built-ins (valid/invalid programs, both backends, bad backend name,
  JSON errors, thin-wall DFM error, file write), permission denial,
  agent-bridge round-trips.

### Evidence (measured)
| Metric | Value |
|---|---|
| Full suite passed | 2431 (was 2401 after M6) |
| Full suite failed | 1 (unchanged: pre-existing M9) |
| Tools suite | 30/30 |
| ruff check / format | clean (714 files) |
| mypy src | 466 files, no issues |
| `validate_program` (100 val programs) | 100/100 all_passed, 0.1 ms/call |
| `execute_program` (20 programs, freecad analytic) | all ok, <1 ms/call |
| Agent `tool_call` round-trip | ok=True, 0.3 ms |

### Next
- M8: self-correction — fix the `correct()` best_risk bug; test; measure.

---

## M8 — Self-Correction

### Goal
Make the bounded self-correction loop (`SelfCorrectingInference.correct`)
actually return its best result: fix the best-result tracking bugs found in
the baseline audit, add a real test suite, and measure on live model output.

### Bugs found and fixed (`src/cadgenesis/inference/self_correction.py`)
1. **Invalid fallback shadowed valid results** — `best_risk` was only updated
   on the invalid branch, so a found valid program could be overwritten by a
   later lower-risk *invalid* attempt.
2. **`return best_result or fallback` discarded invalid results** —
   `SelfCorrectionResult.__bool__` follows `success`, so any invalid
   best-result was falsy and got replaced by the generic "max attempts
   exceeded" fallback (the measured least-risk fallback was never returned).

Fixed: the invalid branch now requires `not best_result.success` before
overwriting (valid always outranks invalid; lowest risk wins within each
class), and the final return checks identity (`best_result is not None`)
instead of truthiness.

### Deliverables
- `tests/inference/test_self_correction.py` (10 tests): scripted
  valid/repair/invalid sequences, least-risk invalid fallback, the two
  regression tests (valid never shadowed by lower-risk invalid; lowest-risk
  valid wins), budget enforcement, real analytic-validator happy/negative
  paths, risk heuristics.

### Evidence (measured)
| Metric | Value |
|---|---|
| Full suite passed | 2441 (was 2431 after M7) |
| Full suite failed | 1 (unchanged: pre-existing M9) |
| Self-correction suite | 10/10 |
| ruff check / format | clean (715 files) |
| mypy src | 466 files, no issues |
| Raw greedy output valid (50 val prompts, M4-xattn-frozen) | 34/50 |
| Non-empty invalid outputs corrected to valid | 4/5 |
| Empty outputs (uncorrectable) | 11/50 |
| Loop latency (3 attempts) | 38.1 ms |

### Next
- M9: evaluation — fix `test_dimension_relative_error`; full eval family;
  measure.

---

## M9 — Evaluation

### Goal
Close out the final failing test from the baseline (the only red test since
M4) and measure the trained model with the evaluation family.

### Findings and fixes
1. **`test_dimension_relative_error` expected a miscalculation** — the metric
   (`GeometryMetrics.dimension_relative_error`) divides by the *reference*
   dimension (standard relative error, documented); the test's hand-computed
   0.125 used the *predicted* denominator (5/20 instead of 5/25).  The
   metric was correct; the test expectation was fixed (0.1) and coverage
   extended with a zero-reference case.
2. **Zero-reference pathological value** — the `max(|r|, eps)` guard made a
   zero reference yield ~1e6 error.  Fixed in
   `src/cadgenesis/evaluation/geometry_metrics.py`: zero reference now means
   error 1.0 when the prediction is non-zero, 0.0 when both are zero
   (bounded in [0, 1]).

### Evidence (measured)
| Metric | Value |
|---|---|
| Full suite passed | **2442 (0 failed — first fully green run since baseline)** |
| Evaluation suite | 69/69 |
| ruff check / format | clean (715 files) |
| mypy src | 466 files, no issues |
| Greedy output on 200 val prompts (M4-xattn-frozen) | parse 200/200, IR-valid 161/200 (80.5%) |
| mean bbox_iou vs ground truth (200 pairs) | 0.800 |
| mean dimension_relative_error vs ground truth | 0.160 |
| TOON-path validity via execution | 0.000 on raw token streams (metric is TOON-string based; canonical validity for token programs is the IR validator — see IR-valid above) |

### Next
- M10: deployment — fix `cli/deploy.py` remote list 404 and stand up the
  FastAPI/uvicorn serving path; verify.