# Pillar 7 — Neuro-Symbolic Reasoning

Implementation report for the **Neuro-Symbolic Reasoning** pillar of the
CADGenesis-LM v6.0 roadmap (`docs/v6_roadmap.md`). Closes the capability and
integration gaps identified in `docs/pillar7_reasoning_audit.md` — all
**additive** and backward compatible. The `reasoning/` package was already
fully implemented (rules, constraints, geometry, topology, symbolic reasoner,
neuro-symbolic engine: 2,434 lines, 143 tests); this milestone adds the
missing reasoning capabilities, the `knowledge_network/` package, an
evaluation suite, a benchmark, and one integration bug fix.

## 1. Scope (requirements → modules)

| # | Capability | Module |
|---|-----------|--------|
| 1 | Backward chaining (goal-directed rule proofs) | `reasoning/rule_engine.py` (`prove`, `prove_all`, `Proof`) |
| 2 | Rule versioning (snapshot / diff / by-version) | `reasoning/rule_engine.py` (`Rule.version`, `snapshot`, `by_version`, `diff`) |
| 3 | Engineering standards library (ISO/ASME/DIN/ANSI/company) | `reasoning/standards.py` (`StandardsLibrary`, 11 defaults, ISO 286 / 261 / 1302, ASME B4.1, Y14.5, DIN, ANSI, `build_standards_graph`) |
| 4 | Casting / welding / tooling / tolerance manufacturability checks | `reasoning/manufacturing_rules.py` (`check_casting`, `check_welding`, `check_tooling`, `check_tolerance`) |
| 5 | Symbolic planner (best-first cost-ordered planning) | `reasoning/symbolic_planner.py` (`SymbolicPlanner`, `PlanningOperator`, `SymbolicPlan`) |
| 6 | Feature-dependency / topology adjacency reasoning | `reasoning/geometry_reasoner.py`, `reasoning/topology.py` |
| 7 | Hybrid reasoning pipeline (neural + knowledge + rules + constraints + geometry + manufacturing, refinement, explanation) | `reasoning/hybrid.py` (`HybridReasoningPipeline`, `HybridReasoningReport`, `StageReport`) |
| 8 | Constraint propagation / conflict detection / repair | `reasoning/constraint_solver.py` (`propagate`, `detect_conflicts`, `repair`) |
| 9 | Knowledge network package (multi-source search/lookup) | `knowledge_network/` (`KnowledgeNetwork`, `KnowledgeGraphSource`, `StandardsSource`) |
| 10 | Reasoning evaluation metrics (was a stub) | `evaluation/reasoning_metrics.py` (8 metrics + `run_reasoning_benchmark`) |
| 11 | Benchmark | `benchmarks/reasoning_benchmarks.py` (6 sections) |
| 12 | Integration bug fix: `forward()` missing on the neural engine | `reasoning/neuro_symbolic.py` (fixes `agents/integration.py::NeuroSymbolicAdapter`) |

## 2. Architecture

```
reasoning/                        reasoning/standards.py ──► build_standards_graph() ──► knowledge_graph.KnowledgeGraph
 ├── rule_engine.py               # Rule.version, Proof, prove/prove_all (backward chaining)
 ├── symbolic_planner.py          # SymbolicPlanner (BFS, state hashing, cost ordering) ──► CADPlan
 ├── hybrid.py                    # HybridReasoningPipeline: neural → knowledge → rules →
 │                                #   constraints → geometry → manufacturing → custom → refinement
 ├── manufacturing_rules.py       # casting / welding / tooling / tolerance checks
 ├── constraint_solver.py         # propagate / detect_conflicts / repair
 ├── geometry_reasoner.py         # feature_order / geometric_consistency / tolerance_stack
 ├── topology.py                  # adjacency_graph / connectivity_reasoning
 └── neuro_symbolic.py            # forward(symbolic_facts, neural_state) ──► (corrected, scores)
knowledge_network/
 ├── network.py                   # KnowledgeNetwork facade
 └── sources.py                   # KnowledgeGraphSource, StandardsSource
evaluation/reasoning_metrics.py   # 8 metrics + run_reasoning_benchmark
benchmarks/reasoning_benchmarks.py
```

All additions compose **on top of** the existing engines; no existing public
API was changed or removed.

## 3. Key APIs

| Component | API |
|-----------|-----|
| `RuleEngine` | `prove(goal, context, depth_limit=8) -> Proof`, `prove_all(goals, context, depth_limit)`, `snapshot()`, `by_version(version)`, `diff(other)`; `Rule.version`, `Rule.concludes()`, `Rule.requires()`; `make_rule(..., version=)` |
| `StandardsLibrary` | `register/get/by_body/by_kind/identifiers/bodies`, `tolerance(nominal_mm, grade)` (ISO 286), `fit(symbol)` (ASME B4.1), `roughness_grade(symbol)` (ISO 1302), `thread_pitch(designation)` (ISO 261), `material(name)`, `compliance(part)`, `passed(part)`, `summary()`; `default_standards_library()` (11 standards), `build_standards_graph()` |
| `ManufacturingRules` | `check_casting(part)`, `check_welding(part)`, `check_tooling(part)`, `check_tolerance(part)`; `assess` accepts processes `casting`/`welding`/`tooling`/`tolerance` |
| `SymbolicPlanner` | `plan(goal, initial, is_goal, max_depth=8, max_states=1000) -> SymbolicPlan`, `decompose(goal, is_goal, initial)`, `register/get/operator_names`; `PlanningOperator(name, precondition, effect, cost, action)` |
| `SymbolicPlan` | `solved`, `dependency_graph()`, `execution_order()`, `to_cad_plan(goal)`, `summary()` |
| `HybridReasoningPipeline` | `reason(context, neural_hidden=None) -> HybridReasoningReport`, `add_stage(name, predicate)`; `STAGE_ORDER` |
| `HybridReasoningReport` | `stage(name)`, `stage_names()`, `summary()`, `explain()`; `score`, `passed`, `blocked`, `refined` |
| `ConstraintSolver` | `dependency_graph(constraints)`, `propagate(variables, constraints, assignment, max_hops=4)`, `detect_conflicts(variables, constraints)`, `repair(variables, constraints, relax_order=None)` |
| `GeometryReasoner` | `feature_dependencies(features)`, `validate_feature_dependencies(features)`, `feature_order(features)`, `geometric_consistency(primitives, allowed_interference=0.0)`, `tolerance_stack(chain)` |
| `TopologyAnalyzer` | `adjacency_graph(faces)`, `connectivity_reasoning(faces)` |
| `KnowledgeNetwork` | `register(source)/unregister(name)`, `search(query, source=None, top_k)`, `lookup(key, source=None)`, `all(source=None)`, `to_graph()`, `stats()`; `KnowledgeGraphSource`, `StandardsSource` |
| `NeuroSymbolicReasoningEngine` | `forward(symbolic_facts, neural_state) -> (corrected, scores)` (new; `evaluate_constraints` unchanged) |
| `evaluation/reasoning_metrics.py` | `reasoning_accuracy`, `symbolic_consistency`, `rule_utilization`, `engineering_correctness`, `manufacturing_correctness`, `constraint_reasoning`, `topology_reasoning`, `run_reasoning_benchmark` |

### Hybrid pipeline semantics

- Stages run in `STAGE_ORDER`: `neural` (optional, non-critical), `knowledge`
  (optional, non-critical), `rules`, `constraints`, `geometry`,
  `manufacturing`, then any registered custom stages (critical by default).
- `report.blocked` is set when any **critical** stage fails; a blocked report
  is always rejected.
- `score` = neural symbolic score (when a neural engine is configured) × the
  fraction of critical stages passing.
- **Refinement**: when nothing critical failed but `0.4 <= score < threshold`
  and a neural engine is present, the score is nudged toward the threshold
  (`score + 0.1 * (1 - score)`), the decision is re-evaluated, and a
  `refinement` stage is recorded.
- Rules block the decision only at `severity_index() >= 2` (error/critical);
  warnings are reported but do not block.
- `prove()` evaluates rule conditions against the context **merged with facts
  derived during the search**; the caller's context is never mutated.

## 4. Behavior notes (backward compatibility)

- `ManufacturingRules.assess` now accepts `casting`, `welding`, `tooling` and
  `tolerance` as processes (previously unknown processes raised). The legacy
  test `tests/reasoning/test_manufacturing_rules.py::TestAssess::test_unknown_process`
  was updated to use `["quantum_machining"]`; genuine unknown processes are
  still rejected.
- `NeuroSymbolicAdapter.reason()` in `agents/integration.py` was already
  calling `forward(symbolic_facts, neural_state)`; the engine now implements
  it, so the adapter works without modification.

## 5. Verification

- **Tests**: 106 new tests (`tests/reasoning/test_rule_backward_chaining.py`,
  `test_standards.py`, `test_manufacturing_extended.py`,
  `test_symbolic_planner.py`, `test_hybrid_pipeline.py`,
  `test_constraint_repair.py`, `test_geometry_topology_extended.py`,
  `tests/knowledge_network/test_knowledge_network.py`,
  `tests/evaluation/test_reasoning_metrics.py`). Full suite:
  **1491 passed** (was 1385 before the pillar).
- **Lint**: `ruff check` clean on `reasoning/`, `knowledge_network/`,
  `evaluation/reasoning_metrics.py`, `evaluation/__init__.py`,
  `benchmarks/reasoning_benchmarks.py`.
- **Types**: `mypy` clean (15 source files).
- **Benchmark** (`benchmarks/reasoning_benchmarks.py --reps 2`): rules
  evaluate/backward-prove ~0.016 ms, graph search 0.003 ms, constraint solve
  0.021 ms / propagate 0.010 ms / conflicts 0.019 ms / repair 0.020 ms,
  standards tolerance lookup 0.001 ms, planner plan 0.020 ms, hybrid reasoning
  0.035 ms.
- **Audit**: `scripts/audit_repo.py` reports Neuro-Symbolic Reasoning and
  Knowledge Network pillar modules `[OK]` (the `knowledge_network/` package is
  now implemented and tracked; see `docs/pillar7_reasoning_audit.md` for the
  before/after gap table).
