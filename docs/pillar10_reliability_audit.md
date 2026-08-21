# Pillar 10 — Reliability & Confidence AI: Repository Audit

Audit performed before implementation (v6.0 roadmap, Pillar 10).

## 1. Implemented reliability components

| Module | Status | Notes |
|---|---|---|
| `transformer/heads.py` | Implemented | `ConfidenceHead` (d_model → 1 logit), `LMHead`, `OutputHeads`. |
| `transformer/losses.py` | Implemented | `ConfidenceLoss` (BCE vs. correctness), `CADSequenceLoss` (confidence_weight=0.1). |
| `transformer/uncertainty_attention.py` | Implemented | `UncertaintyAttention` — attention head emitting per-token confidence logits. |
| `transformer/dynamic_routing.py` | Implemented | `EarlyExitGate(threshold)` — confidence-triggered early exit (computation routing). |
| `transformer/geometry_transformer.py` etc. | Implemented | Forward contract `(logits, confidence)` across `GeometryAwareTransformer`, `HierarchicalCADTransformer`, `SelfDesigningTransformer`. |
| `inference/engine.py` | Implemented | `CADInferenceEngine` (greedy/beam/batch) — records per-token + mean confidence passively; never gates on it. |
| `confidence/confidence_engine.py` | Partial | `ConfidenceEngine.compute_sequence_confidence(logits, confidence_head_output)` — entropy+head blend. Static, torch-hard. |
| `memory/memory_router.py` | Implemented | `route_by_confidence(query, confidence, low_pool, high_pool)` — memory routing by confidence. |
| `monitoring/` | Implemented | Health checks, PSI/KL drift (`compute_drift`, `FeatureDriftMonitor`), alerts (`AlertManager`) — infrastructure only. |
| `telemetry/` | Implemented | Metrics registry, structured logs, tracing. |
| `reasoning/symbolic_reasoner.py` | Implemented | `check_constraint`, `check_implication`, `check_token_consistency` — symbolic verification substrate. |
| `execution/execution_engine.py` | Implemented | Geometry validity, `_apply_repair` (mesh repair), `compute_confidence` heuristic. |
| `cad/mesh/repair.py` | Implemented | `diagnose`, `fill_holes`, `remove_duplicate_vertices`. |
| `world_model/functional.py`, `assembly.py` | Implemented | `FunctionalReasoner`, assembly validation substrate. |

## 2. Stubs (docstring-only) — `confidence/` package

`confidence.py` (token/sequence scoring), `uncertainty.py` (epistemic/aleatoric/Bayesian), `calibration.py` (temperature/Platt/isotonic/ECE), `risk.py`, `monitoring.py`, `fallback.py` — **6 stubs**, none imported anywhere.

`confidence/__init__.py` exports only `ConfidenceEngine`.

## 3. Missing capabilities (vs. mission)

- **Confidence estimation**: token-level, geometry/engineering/manufacturing confidence — absent.
- **Uncertainty**: epistemic vs aleatoric decomposition, Bayesian approximation (MC-dropout), ensemble uncertainty — absent repo-wide.
- **Calibration**: temperature scaling, isotonic regression, reliability diagrams, ECE — absent repo-wide.
- **Hallucination detection** (invalid CAD, impossible geometry, broken assemblies, invalid constraints, unsupported operations) — absent entirely.
- **Automatic verification** (CAD validity, engineering correctness, symbolic consistency, memory consistency, planning consistency) — substrate exists (execution, symbolic reasoner, world model) but no orchestrator, and nothing is confidence-gated.
- **Automatic repair** (geometry/topology/constraints/planning/reasoning) — mesh repair exists in `cad/mesh/repair.py` + engine `_apply_repair`; no orchestrated repair layer.
- **Confidence-aware routing** — only memory routing (`route_by_confidence`) and computation early-exit; no retrieval/symbolic/planner/expert routing.
- **Dynamic fallback** (expert escalation, retrieval augmentation, symbolic verification, multi-agent verification) — absent.
- **Risk assessment** (design/manufacturing/simulation/safety) — absent.
- **Explainability** (confidence/uncertainty reports, reasoning trace, validation report) — absent.
- **Confidence-driven inference pipeline** — inference records confidence but never acts on it (no gating, no repair, no rerouting).
- Torch guard: `confidence/__init__.py` hard-imports torch.

## 4. Duplicated functionality

1. Confidence blending: `ConfidenceEngine.compute_sequence_confidence` vs ad-hoc sigmoid means in `inference/engine.py` — should be unified.
2. Validity checks: execution validator + symbolic reasoner + world model overlap; no single verification orchestrator.
3. Drift monitoring exists in `monitoring/` but nothing monitors confidence distribution drift.

## 5. Architecture plan (backward compatible)

1. **`confidence/`** — fill 6 stubs and add: `hallucination.py`, `verification.py`, `repair.py`, `routing.py`, `report.py`, `pipeline.py`.
2. **Confidence**: token + sequence + geometry + engineering + manufacturing estimators (pure Python, torch-optional).
3. **Uncertainty**: epistemic (MC-dropout), aleatoric (predictive entropy), Bayesian approximation, ensemble disagreement.
4. **Calibration**: temperature scaling, isotonic (PAV), reliability diagrams, ECE; `CalibratedConfidenceEngine` wrapper.
5. **Hallucination detection**: layered checks over CAD execution, world model, symbolic reasoner, tokenizer vocabulary.
6. **Verification**: `AutomaticVerifier` (CAD validity, engineering, symbolic, memory, planning).
7. **Repair**: `AutomaticRepair` (geometry via mesh repair, topology, constraints, planning, reasoning) with fallback ladder.
8. **Routing**: retrieval/memory/symbolic/planner/expert confidence-aware routers + router registry.
9. **Fallback**: dynamic strategy chain (expert escalation → retrieval → symbolic → multi-agent).
10. **Risk + explainability**: per-domain risk scores; structured confidence/uncertainty/validation reports and reasoning traces.
11. **`ReliabilityPipeline`**: Prompt → Inference → Confidence → Uncertainty → Verification → Repair → Validation → Confidence Routing → Final Output, wrapping any inference callable (works with `CADInferenceEngine` without modifying it).
12. Integrations: agents (ConfidenceAdapter extension), memory, world model, execution, learning system.
13. Tests under `tests/confidence/`, benchmarks, docs.

Nothing existing is removed; `ConfidenceEngine.compute_sequence_confidence` is preserved; all additions are additive.
