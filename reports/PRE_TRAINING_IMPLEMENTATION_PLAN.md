# PHASE 0 — Repository Audit & Prioritized Pre-Training Implementation Plan

Generated: 2026-08-20
Gate status: TRAINING BLOCKED — this plan must be executed before the PRE-TRAINING READINESS REVIEW.

---

## 1. Audit Method

- Inventory of `src/cadgenesis/` (45 subpackages, 404 modules), `tests/` (187 test files), `scripts/`, `benchmarks/`, `checkpoints/`, `data/`, `configs/`, `experiments/`.
- Checklist A–J from the pre-training gate mapped to concrete modules, classes, and tests.
- Stub detection: modules with substantive implementations (0 stubs per `scripts/audit_repo.py`).

## 2. Executive Findings

### 2.1 What already exists (substantive, verified)

| Checklist Area | Verdict | Evidence |
|---|---|---|
| **A. Model** | ✅ PASS | `config/cad_config.py`: `ModelConfig` (d_model=1024, nhead=16, 12 enc + 12 dec layers, dim_feedforward=4096, max_seq_len=2048, vocab slot families, MoE/SSM/MLA/GQA options, precision via `TrainingConfig.mixed_precision="bf16"`, schedule cosine/WSD). Presets mini→large (`from_preset`). Objective: `MaskedCrossEntropyLoss`/`CADSequenceLoss` (`transformer/losses.py`). Trainer: `training/trainer.py` + `optimizer.py`, `scheduler.py`, `checkpoint.py` (CheckpointManager, meta.json, resume), `callbacks.py` (checkpoint/early-stop/metrics-logging), `fsdp.py`, `distributed.py`, `deepspeed.py`, `packing.py`, `metrics.py`, `profiler.py`, `mu_transfer.py`. Real checkpoints exist in `checkpoints/` (run1/run2/m4-curriculum/teach-1.5b-smoke at 9.2 GB). |
| **B. Dataset** | 🟡 PARTIAL | `datasets/cad_jsonl.py`: `CADJsonlDataset`, `load_jsonl`, minhash dedup. `datasets/curriculum.py`: `quality_filter`, `adversarial_records`, deterministic type-stratified leakage-free `make_splits`, manifest with per-split digest. `datasets/cad_program_synth.py`: synthetic pipeline. Real data: `data/curriculum/` (train 8002 / val 999 / test 999), `data/cad_programs.jsonl`. |
| **C. Tokenization** | ✅ PASS | `tokenizer/cad_tokenizer.py`, `vocabulary.py`, `serialization.py`, `toon_backend.py`; token families: geometry/feature/constraint/material/assembly/manufacturing/simulation/numeric/language. Round-trip tests pass: `tests/tokenizer/test_serialization.py`, `test_cad_tokenizer.py`, `test_toon_backend.py`, `test_compression.py`. |
| **D. Tool Calling** | 🟡 PARTIAL | `tools/schema.py` (ParameterSpec, ToolDefinition, ToolCall, ToolResult), `tools/registry.py` (permission-gated validation), `tools/executor.py` (`ToolExecutor` with 5 built-in CAD tools: validate_program, execute_program, analyze_brep, estimate_cost, manufacturing_check, export_program), `tools/agent.py`. |
| **E. CAD Execution** | ✅ PASS | `execution/execution_engine.py` (intent → program → execute → validate → simulate → optimize → repair → export → feedback), `freecad_engine.py`, `opencascade_engine.py` + analytic fallback, `ir_execution.py`, `exporter.py`, `topology_analysis.py`, `manufacturing.py`, `simulation.py`, `cost_estimation.py`, `optimization.py`, `feedback.py`. |
| **F. Validation** | ✅ PASS | `execution/geometry_validation.py` (GeometryCheck/Report, analytic + triangle-intersection tests), `topology_analysis.py` (manifold/closure/Euler/genus), IR validation (`validate_program_ir`), `cad/mesh/repair.py` (`diagnose`). |
| **G. Self-Correction** | 🟡 PARTIAL | `inference/self_correction.py` (bounded retry loop: validate → identify failure → repair → retry), `reasoning/constraint_solver.py` (`repair()`), `execution/execution_engine._apply_repair` (mesh repair with report), `evaluation/execution_metrics.py` (`repair_rate`). |
| **H. Benchmarking** | 🟡 PARTIAL | `evaluation/benchmark_runner.py`, `cad_bench.py`, metric modules (cad/geometry/execution/reasoning/tokenizer/world_model/agent), `benchmarks/` (9 scripts, currently micro-benchmarks). |
| **I. Confidence** | 🟡 PARTIAL | `confidence/calibration.py`: TemperatureScaling, PlattScaling, `ConfidenceCalibrator`, **ECE implemented**; `confidence.py` (ConfidenceEngine), `uncertainty.py`, `risk.py`, `monitoring.py`, `fallback.py`. |
| **J. Reproducibility** | 🟡 PARTIAL | `experiments/` registry (meta.json with hyperparams/metrics), `research/experiments.py`, config round-trip tests (`tests/config/test_config_roundtrip.py`). |

### 2.2 Verified gaps (the real work)

| # | Gap | Area | Severity | Evidence |
|---|---|---|---|---|
| G1 | **Brier score not implemented** | I. Confidence | HIGH | **[DONE]** `brier_score` in calibration.py — 15/15 tests |
| G2 | **Reliability diagram not implemented** | I. Confidence | MED | **[DONE]** `reliability_diagram` in calibration.py — 15/15 tests |
| G3 | **Abstention mechanism not implemented** | I. Confidence | HIGH | **[DONE]** `AbstentionPolicy` + `ABSTAIN` in fallback.py — 15/15 tests |
| G4 | **Tool timeout handling missing** | D. Tool Calling | HIGH | **[DONE]** `timeout_seconds` + watchdog in executor.py — 12/12 tests |
| G5 | **Per-call tool provenance missing** | D. Tool Calling | MED | **[DONE]** call_id/caller/run_id/timestamp/duration — 12/12 tests |
| G6 | **Dataset validator module missing** (per-record schema/quality checks, dedup report, stats report) | B. Dataset | HIGH | **[DONE]** `datasets/validator.py` — 20/20 tests, `reports/DATASET_VALIDATION.md` |
| G7 | **Dataset versioning incomplete** | B. Dataset | MED | **[DONE]** `DatasetRegistry` wired via `ResearchSession.snapshot_dataset`, attached to experiments |
| G8 | **Benchmark dataset (eval set) missing** | H. Benchmarking | HIGH | **[DONE]** `data/benchmarks/eval_set.jsonl` (103 held-out records, seed 999 disjoint from curriculum seed 0) |
| G9 | **Baselines not defined** | H. Benchmarking | HIGH | **[DONE]** `RandomBaseline` + `FrequencyBaseline` + `ProgramOracle` — 11/11 tests |
| G10 | **Evaluation reports not generated** | H. Benchmarking | MED | **[DONE]** `reports/BENCHMARK_REPORT.md` (baselines score 0.0 — model must beat 0.0) |
| G11 | **Failure classification taxonomy missing** | G. Self-Correction | MED | **[DONE]** `execution/failure_modes.py` (10 modes), wired into self_correction.py — 21/21 tests |
| G12 | **Repair metrics incomplete** (initial success rate, iterations-to-success, repair success rate) | G. Self-Correction | MED | **[DONE]** 4 new metrics in execution_metrics.py — 17/17 tests |
| G13 | **Experiment registry under-used** (1 experiment only; no dataset/model/hardware/env tracking) | J. Reproducibility | MED | **[DONE]** `new_experiment` auto-attaches environment + dataset versions — 6/6 tests |
| G14 | **Loss curves not persisted as structured artifacts** | — | MED | **[DONE]** `MetricsJsonlCallback` → metrics.jsonl + `scripts/plot_loss.py` — 5/5 tests |
| G15 | **Staged smoke-test suite for training pipeline missing** (1-batch forward/backward, tiny overfit, etc.) | Smoke | HIGH | **[DONE]** `cadgenesis.smoke` stages 1-4 + `scripts/smoke/run_all.py`; ALL STAGES PASS (overfit 6.26→0.49 < 0.5 in 137 steps); `reports/SMOKE_TEST_RESULTS.md` |

### 2.3 Critical context

- Real training infrastructure AND real prior training runs exist (checkpoints present, scripts/train.py, teach.py, distill_train.py). **No fake checkpoints are needed** — this is genuine infrastructure.
- No 1.5B-scale convergence evidence documented; teach-1.5b-smoke exists but convergence/loss curves are not persisted.
- **Training remains BLOCKED** until gaps G1–G15 are closed and the readiness review is generated.

## 3. Prioritized Implementation Plan (Phase Order)

### PHASE 1 — Foundation Model Specification (docs + config, no training)
1. Write `reports/FOUNDATION_MODEL_SPEC.md` documenting the canonical model: name, architecture type (encoder-decoder, lean default), parameter count for each preset (compute from ModelConfig), layers, hidden dim, heads, context, vocab, precision, objective, CAD representation (TOON token stream).
2. Add `total_parameters()` estimator utility (config → param count) in `config/cad_config.py` or `transformer/`; unit test it.
3. Pin the v8.0 training preset (mini/small/base) with exact hyperparameters in `configs/examples/`.

### PHASE 2 — Dataset Architecture + Validation
1. **G6**: new `src/cadgenesis/datasets/validator.py` — per-record schema validation, token-validity check, quality rules (min/max length, token coverage), dedup report, statistics report (counts per type, token frequency, coverage).
2. **G7**: extend `dataset_manifest.json` generation to include dataset version, generator version, date, and split digests (already partial); add `DatasetRegistry` for version tracking.
3. Tests: `tests/datasets/test_validator.py` — validates `data/curriculum/` and `data/cad_programs.jsonl`.

### PHASE 3 — CAD Representation + Tokenization Validation
1. Run existing tokenizer round-trip suites; fix any failures.
2. Add tokenizer round-trip test over the full `data/curriculum/` corpus (encode → decode → exact match, lossless).
3. Verify constraint/feature/tool-call token families are exercised by tests (`tests/tokenizer/`).

### PHASE 4 — CAD Tool-Calling Protocol
1. **G4**: add `timeout_seconds` to `ToolDefinition` + enforcement in `ToolExecutor.dispatch` (asyncio or thread-based watchdog; errors returned as `ToolResult(ok=False)`).
2. **G5**: add `call_id`, `caller`, `run_id`, `timestamp` to `ToolCall`/`ToolResult`; wire into `tools/registry.py`.
3. Tests: `tests/tools/test_timeout.py`, `test_provenance.py`.

### PHASE 5 — CAD Execution Engine
1. Verify each execution stage on the analytic fallback backends (already real): sketch, constraint, feature, boolean, parametric regeneration, export, failure capture.
2. Add end-to-end pipeline test: NL → program → execute → validate → export for 20 diverse curriculum records.
3. Document backend capability matrix (FreeCAD / OpenCascade / analytic).

### PHASE 6 — Geometry / Constraint / Topology Validation
1. Run `tests/execution/test_validators.py`, `tests/cad/*` suites; fix failures.
2. Add assembly validity + parametric-history validation checks if missing (`cad/assembly` exists; verify coverage).

### PHASE 7 — Failure Detection + Diagnosis
1. **G11**: define failure-mode taxonomy enum (e.g., MISSING_FEATURE, BAD_DIMENSION, NON_MANIFOLD, UNDERCONSTRAINED, INFEASIBLE_CONSTRAINT, UNKNOWN_TOKEN, EXECUTION_ERROR) in `execution/feedback.py` or new `execution/failure_modes.py`; classify failures from validation reports.
2. Tests: classification unit tests per failure mode.

### PHASE 8 — Self-Correction System
1. **G12**: extend `evaluation/execution_metrics.py` with `initial_success_rate`, `repair_success_rate`, `iterations_to_success`; compute from `self_correction.py` loop.
2. Add iteration-limit + repair-metrics tests: `tests/inference/test_self_correction_metrics.py`.

### PHASE 9 — Benchmark + Evaluation Infrastructure
1. **G8**: create `data/benchmarks/` eval set (e.g., 100 held-out curriculum-style records + 20 hand-authored NL→CAD prompts), documented.
2. **G9**: define baselines (random-token baseline, n-gram baseline) in `evaluation/cad_bench.py`.
3. **G10**: `evaluation/report_generator.py` → produce `reports/BENCHMARK_REPORT_*.md` (metrics: NL→CAD, constraint satisfaction, feature/parameter accuracy, geometric/topology validity, design-intent accuracy, repair success).
4. Tests: `tests/evaluation/test_cad_bench.py`.

### PHASE 10 — Confidence Calibration Infrastructure
1. **G1**: implement `brier_score` in `confidence/calibration.py` (multiclass Brier).
2. **G2**: implement `reliability_diagram` (bin confidences → accuracy curve data points).
3. **G3**: implement abstention: `ConfidenceEngine.compute_sequence_confidence` returns `(confidence, calibrated)`, add `abstain(confidence, threshold)` policy + `AbstentionPolicy` in `confidence/fallback.py`.
4. Tests: `tests/confidence/test_calibration_metrics.py` (ECE, Brier, reliability, abstention) on synthetic logits.

### PHASE 11 — Reproducibility + Experiment Infrastructure
1. **G13**: extend `experiments/` registry: snapshot dataset version, model config hash, hardware (torch.cuda), software versions (torch/python), seed → stored in meta.json on run start.
2. **G14**: metrics logging callback writes `metrics.jsonl` (step, loss, val_loss, lr, throughput) alongside checkpoints; `scripts/plot_loss.py` to render loss curves from it.
3. Tests: config round-trip already exists; add registry round-trip test.

### PHASE 12 — Small-Scale Smoke Tests (CPU)
1. **G15**: `scripts/smoke/stage1_forward_backward.py` — 1 batch forward/backward on mini preset. **[DONE]**
2. `scripts/smoke/stage2_tiny_dataset.py` — train 50 records, 1 epoch, CPU. **[DONE]**
3. `scripts/smoke/stage3_overfit.py` — overfit 8 records to near-zero loss (proves learning). **[DONE]** 6.26 → 0.49 in 137 steps.
4. `scripts/smoke/stage4_dev_run.py` — 200 records, few epochs, record loss curve + metrics.jsonl. **[DONE]**
5. All runnable on CPU in < 5 minutes each; results recorded in `reports/SMOKE_TEST_RESULTS.md`. **[DONE — all stages PASS]**

### FINAL — PRE-TRAINING READINESS REVIEW
1. Run full test suite + all smoke stages.
2. Generate `reports/PRE_TRAINING_READINESS_REVIEW.md` with 14 required sections.
3. Decision: GO or NO-GO. Training must NOT start until this review is written and approved.

## 4. Severity Summary

| Severity | Items |
|---|---|
| HIGH (blocking) | G1 Brier, G3 abstention, G4 tool timeout, G6 dataset validator, G8 benchmark eval set, G9 baselines, G15 smoke suite |
| MED | G2 reliability diagram, G5 tool provenance, G7 dataset versioning, G10 eval reports, G11 failure taxonomy, G12 repair metrics, G13 experiment registry, G14 loss-curve persistence |
| LOW | G0 (none beyond) |

## 5. Rules of Engagement

- NO actual pretraining / fine-tuning / RL / GPU runs until the readiness review passes.
- All phases produce code + tests; every phase runs `python -m pytest` on its tests.
- All phases may produce configs, loaders, tokenizer code, tiny synthetic data, and CPU smoke runs (allowed per gate).
- If any phase reveals a blocker, STOP, fix, re-test, and update this plan before proceeding.