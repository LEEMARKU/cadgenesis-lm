# CADGenesis-LM v6.0 — Upgrade Baseline

**Generated:** 2026-08-18 · **Method:** measured by execution (scripts run against the live repo; no values invented). Values that could not be measured are marked `NOT MEASURED`.

---

## 1. Architecture (from code)

- Encoder–decoder **`GeometryAwareTransformer`** (`src/cadgenesis/transformer/geometry_transformer.py`) with type-embeddings for multimodal CAD token families, RoPE positional encoding, KV-cache decode (`decode_step`, `prepare_decoder_cache`), confidence head, optional MoE/SSM/multi-scale/specialized attention (all off by default).
- Modular transformer family: `transformer/` (34 modules), tokenizer (34 modules), execution (13 modules), confidence (4), inference (4: engine, self-correction, MCTS, EAGLE), memory (19), agents (20), evaluation (14), reasoning (13), continual_learning, optimization, serving, platform, research.
- Full pipeline: NL → CAD token stream → `CADExecutionEngine.execute` → geometry/topology/DFM/cost/simulation validation → feedback loop.

## 2. Model configurations (measured)

| Preset | d_model | enc/dec layers | heads | FFN | block_size | Parameters |
|-------:|--------:|--------------:|------:|----:|-----------:|-----------:|
| mini (`CADConfig.mini()`) | 128 | 3 / 3 | 4 | 512 | 64 | **2,966,306** |
| nano | 128 | 3 / 3 | 4 | 512 | 64 | **6,606,233** |
| small | 384 | 6 / 6 | 8 | 1536 | 64 | **48,448,945** |
| default | 1024 | 12 / 12 | 16 | 4096 | 64 | **565,056,753** |
| 1.5b | 1536 | 16 / 16 | 16 | 6144 | 64 | NOT MEASURED (would OOM) |
| large | 1536 | 24 / 24 | 16 | 6144 | 64 | NOT MEASURED (would OOM) |

- Model output vocab (LM head): **3,392** (sum of token-family slot capacities: 64+1024+512+512+256×4+256).
- RoPE: `rope_theta=10000`, `qk_rope_head_dim=64`, no scaling (`max_position_embeddings=2048`).
- Weight tying: NOT MEASURED (not verified). Dropout 0.1. `attention_backend='math'`.
- mini forward: batch 2×8×8 → logits `[2, 8, 3392]` in **0.093 s CPU**.

## 3. Tokenizer (measured)

- `AutonomousCADTokenizer.build()` → **1,319 registered tokens** across 9 families: SPECIAL 23/64, NUMERIC 744/1024, GEOMETRY 95/512, FEATURE 85/512, CONSTRAINT 75/256, MATERIAL 86/256, ASSEMBLY 59/256, MANUFACTURING 82/256, SIMULATION (measured) 62/256, LANGUAGE **2**.
- **Deficiency:** language side uses `LegacyWordTokenizer` with **only 2 tokens** — natural language is effectively untokenized without `build_lang_vocab(texts)`.
- `build_mini()` → 48 tokens (20 legacy NUM_ bins + 5 primitives + specials).
- Numeric: 256-angle-bin / 1024-length-bin quantization (`NUM_`/`ANG_`), `param_min=0.0`, `param_max=1000.0`.
- No BPE/tokenizer training; no round-trip tests; no engineering-notation (Ø25, R12.5, M8x1.25, ±0.02) tokens verified.

## 4. Dataset (measured)

- `build_synthetic_records(500, seed=42)` → **500 records**, **464 unique prompts**, **51 unique program tokens**, built in <0.1 s.
- **38/51 program tokens are NOT in the default vocabulary** (missing `EXTRUDE`, `BOX`, `CYLINDER`, `SKETCH_RECT`, all `NUM_xx` ≥ 5, `HOLE`, `THREAD`, `PATTERN`, …) — dataset is not tokenizable with the default tokenizer today.
- No train/val/test split artifacts; `cli/train.py` regenerates synthetic data at run time (`--train-size 800`, `--valid-size 200`).
- Analytic validator pass rate on generated programs: **500/500 (100%)**.
- Template count: 18; 4 grammars (FILLET/BOLT/WEIGHT/SHAFT etc.) are rejected by the validator by design.
- Deduplication: none. Quality scoring: none. Leakage checking: none.

## 5. Training pipeline

- Entry: `cadgenesis.cli.train` (mini/full modes), `training/trainer.py` `CADTrainer` + `MultiModalCADDataset` + `cad_collate_fn`.
- Config (mini): batch 64, grad_accum 4, max_epochs 8, lr 3e-4, weight_decay 0.01, grad clip 1.0, warmup 2000, cosine schedule, mixed precision `no` (default config: `bf16`), save every 500 steps, eval every 200 steps, no FSDP/DDP, no packing.
- Checkpoint resume: supported via `--resume-from` (test `test_resume_replays_exact_interrupted_trajectory` currently FAILING).
- Experiment tracking / config hashing: not verified. Early stopping: not verified.
- Actual trained checkpoints: **none exist** (`NOT MEASURED` — no training run performed; 4 GB VRAM constraint).

## 6. Inference / generation

- `decode`/`decode_step` with KV cache; `prepare_decoder_cache`; attention `forward_cached` paths across 6 attention variants.
- `tests/inference/test_kv_cache.py::test_decode_step_matches_full_forward` passes (MLAs).
- Generation settings, latency, memory: NOT MEASURED (no production-level benchmark).
- Streaming: WebSocket endpoint exists in serving layer (unrun).

## 7. CAD execution pipeline

- `CADExecutionEngine`: token prefix evaluation + full pipeline (`execute`) with geometry, topology, manufacturing, simulation, optimization, cost, feedback, export.
- `GeometryValidator.validate_mesh`: 4 analytic checks (watertight, boundary edges, self-intersection, degenerate faces) — implemented in prior session; `tests/execution/` **114/114**.
- FreeCAD/OpenCascade backends: modules exist; availability: NOT MEASURED (no FreeCAD in env).

## 8. Self-correction

- `SelfCorrectingInference.correct()` bounded loop (max 5). **Known regression reproduced by code trace:** a valid result on attempt N can be *overwritten* by an invalid later attempt because `best_risk` is never updated on success and repair runs on already-valid tokens. Manual trace confirms success path exists; loop logic is defective.

## 9. RAG — **NOT IMPLEMENTED** (no retrieval subsystem; knowledge_network/ exists but no embedding store/retrieval pipeline verified)

## 10. Tool calling — **NOT IMPLEMENTED** (agents/ has multi-agent infra; no CAD tool registry with schemas/permissions/executor)

## 11. Evaluation

- `evaluation/`: cad_bench, cad_metrics, agent_metrics, benchmark_runner, geometry_metrics, report_generator — exist. `tests/evaluation/` has 2 failing tests. Live benchmark runs: NOT MEASURED (docs define plan; no executed run log).

## 12. API / deployment

- FastAPI app (`serving/api.py`) with `/api/v1` + `/healthz` `/readyz` `/metrics` + WebSocket; **FastAPI/uvicorn not installed** → `app is None`; live serve untested.
- `cli/deploy.py:122` remote `list` → `POST /api/v1/registry/models/list` vs server `GET /api/v1/registry/models/{name}` → **404 bug (unfixed)**.
- Docker: `docker/Dockerfile` (train), `docker/Dockerfile.serve` exist.
- SDK: `LocalBackend` verified (16 tokens, conf 0.578, CPU); `RestBackend` unrun.

## 13. Observability

- HealthChecker/HealthAggregator/PrometheusExporter verified; no OpenTelemetry/tracing; stdlib logging; no structured JSON logs; no request IDs/trace IDs.

## 14. Security

- RBAC wildcard bug **fixed** (fnmatch; 15/15 auth tests). JWT/API-key/OAuth2 verified.
- Open: `torch.load(weights_only=False)`; no prompt-injection layer; default admin creds; no CAD sandboxing for generated programs.

## 15. Test results (measured, this session)

- **2,264 collected → 2,242 passed / 22 failed** (235.8 s), 2 warnings; ruff clean; mypy clean.
- Failures (all pre-existing): adapters 6, distillation 10, evaluation 2, training 1, continual_learning 1.

## 16. Known failing tests (exact)

```
tests/adapters/test_peft.py::test_lora_forward_matches_manual_delta
tests/adapters/test_promotion.py::test_approve_when_meets_thresholds
tests/adapters/test_promotion.py::test_drift_within_tolerance_approved
tests/adapters/test_promotion.py::test_falls_back_to_metadata_scores
tests/adapters/test_promotion.py::test_promote_updates_status_when_approved
tests/adapters/test_promotion.py::test_promote_keeps_status_when_rejected
tests/continual_learning/test_adapter_isolation.py::test_nested_isolations_restore_their_own_baseline
tests/distillation/test_consensus.py::test_toon_votes_majority_winner_and_agreement
tests/distillation/test_consensus.py::test_toon_votes_weights_change_winner
tests/distillation/test_critique.py::test_critique_flags_unparsable_toon
tests/distillation/test_hard_labels.py::test_mask_tokens_ignores_ignore_index_positions
tests/distillation/test_hard_labels.py::test_min_confidence_masks_low_confidence_positions
tests/distillation/test_hard_labels.py::test_masked_labels_are_loss_ready
tests/distillation/test_pipeline.py::test_run_end_to_end_no_network
tests/distillation/test_pipeline.py::test_compute_loss_delegates_to_loss_pipeline
tests/distillation/test_pipeline.py::test_run_honors_custom_temperature_and_alpha
tests/distillation/test_pipeline.py::test_run_zero_samples_is_safe
tests/distillation/test_soft_labels.py::test_kl_loss_is_zero_for_identical_logits
tests/distillation/test_synthetic.py::test_apply_perturbation_zero_noise_is_identity
tests/evaluation/test_geometry_metrics.py::test_dimension_relative_error
tests/evaluation/test_tokenizer_metrics.py::test_vocabulary_coverage_full_tokenizer
tests/training/test_train_script.py::test_resume_replays_exact_interrupted_trajectory
```

## 17. Top 20 deficiencies ranked by impact

| # | Deficiency | Impact | Milestone |
|--:|------------|--------|-----------|
| 1 | 38/51 dataset tokens missing from default vocab | model cannot encode its own dataset | M1 |
| 2 | Language tokenizer has 2 tokens (no NL vocab) | NL side collapses to unk | M1 |
| 3 | Model vocab 3392 ≠ 1319 registered tokens | id space mismatch / unusable embeddings | M1 |
| 4 | Self-correction loop overwrites valid results | repair fails despite valid candidate | M8 |
| 5 | 22 failing tests (adapters 6, distillation 10, eval 2, training 1, CL 1) | CI red | M3/M9 |
| 6 | RAG not implemented | no grounding/retrieval | M6 |
| 7 | Tool calling not implemented | no agent loop | M7 |
| 8 | Dataset tiny (500 records, 1 category) + no splits/dedup | model can't learn | M3 |
| 9 | No CAD-IR (free-form token stream only) | no structured reasoning | M2 |
| 10 | No actual training run / checkpoints | nothing to serve | M4 |
| 11 | FastAPI uninstalled → live serve untested | deployment unverified | M10 |
| 12 | deploy.py remote list 404 | broken client | M10 |
| 13 | No streaming/SSE test, no load test | unverified UX | M10 |
| 14 | No structured logging/tracing/request IDs | hard to debug | M10 |
| 15 | weights_only=False + no sandbox for generated programs | security risk | M10 |
| 16 | No quality gate / benchmark gate for model promotion | unverified claims | M9 |
| 17 | No KV-cache-vs-full equivalence tests for all attention variants | risk in generation | M1 |
| 18 | No tokenizer round-trip/efficiency benchmark | tokenizer unmeasured | M1 |
| 19 | No constraint/topology/intent verification layer | validators shallow | M8/M9 |
| 20 | Ablation experiments never executed (docs only) | no evidence | M9 |

## 18. Implementation roadmap

1. **M1** Transformer + tokenizer: unify vocab (dataset ∩ vocab), BPE language tokenizer or corpus vocab build, round-trip + efficiency tests, KV-cache equivalence tests, NaN/vocab-boundary tests.
2. **M2** CAD-IR: schema (pydantic-style dataclasses), validation, serialization, dependency graph, versioning, tests.
3. **M3** Dataset: multi-category procedural generator (NL→IR, IR→program, program→explanation, error→correction, geometry→description, constraint, parameter, tool, planning), quality filter (syntax→schema→execute→geometry→constraint→dedup→score), splits, adversarial sets.
4. **M4** Training: reproducible mini-config SFT run (CPU/4GB), checkpointing, eval harness, config hash, seed control.
5. **M5** Reasoning: planner + constraint/parameter reasoner over CAD-IR.
6. **M6** RAG: local store (numpy/hash-based), chunking, embedding (hash/bow + optional sentence-transformers), retrieval, precision/recall tests.
7. **M7** Tools: registry + executor over real CADEngine primitives, schemas, permissions, tests.
8. **M8** Self-correction: fix loop bug, structured error types, repair metrics.
9. **M9** Evaluation: benchmark runner with fixed sets, quality gates, ablation execution.
10. **M10** Deployment: install fastapi/uvicorn, fix deploy 404, live serve validation, streaming test, security checks.

After every milestone: `pytest` + `ruff` + `mypy` + project benchmarks, recording before/after in `docs/UPGRADE_LOG.md`.
