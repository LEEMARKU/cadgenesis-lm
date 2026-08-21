# Pillar 7 — Neuro-Symbolic Reasoning: Repository Audit

Audit performed before implementation (v6.0 roadmap, Pillar 7).

## 1. Implemented reasoning modules

`src/cadgenesis/reasoning/` — 11 files, 2 434 lines, **zero stubs**, ruff + mypy
clean, 143 tests green. The roadmap's "Core engine done; rules/KG/planner
stubbed" status is outdated.

| File | Lines | Public API | Status |
|---|---|---|---|
| `rule_engine.py` | 251 | `Rule`, `RuleResult`, `RuleEngine` (`add_rule`, `evaluate`, `run` = forward chaining, `violations`, `summary`), `make_rule`, `SEVERITY_ORDER` | Real (forward-only) |
| `knowledge_graph.py` | 286 | `GraphNode`, `GraphEdge`, `KnowledgeGraph` (add/neighbors/shortest_path/find_related/query, JSON persistence) | Real (adjacency + BFS only) |
| `constraint_solver.py` | 227 | `Variable`, `Constraint`, `Solution`, `ConstraintSolver` (linear, projection-based `solve`, `check_consistency`) | Real (linear numeric only) |
| `geometry_reasoner.py` | 213 | `Primitive`, `GeometryValidation`, `GeometryReasoner` (validate/volume/aabb/overlaps/clearance/contains/fit) | Real (analytic primitives) |
| `manufacturing_rules.py` | 308 | `MfgCheck`, `ManufacturingAssessment`, `ManufacturingRules` (machining, injection molding, 3D printing, sheet metal) | Real (4 processes) |
| `symbolic_reasoner.py` | 228 | `SymbolicExpression` (AST whitelist), `SymbolicReasoner` (evaluate, check_constraint, check_implication, check_token_consistency) | Real |
| `planner.py` | 283 | `PlanningStep`, `CADPlan` (deps/topo order/critical path), `TaskPlanner` (templates + rule refinement) | Real (workflow templates) |
| `topology.py` | 232 | `TopologyStats`, `TopologyAnalyzer` (Euler, genus, manifold/closed, connected components, analyze_mesh) | Real |
| `validator.py` | 285 | `CheckResult`, `ValidationReport`, `DesignValidator` (rule/constraint/geometry/manufacturing/topology/symbolic/custom) | Real |
| `neuro_symbolic.py` | 45 | `NeuroSymbolicReasoningEngine` (`evaluate_constraints`) | Real but minimal — **no `forward`** |
| `__init__.py` | 76 | 27 re-exports | Real |

## 2. Missing reasoning capabilities

| Capability | Gap |
|---|---|
| Rule engine — **backward chaining** | No goal-directed proof, no justification trace |
| Rule **versioning** | No version metadata, no versioned rule sets |
| **Engineering standards** (ISO/ASME/DIN/ANSI/company) | Nothing exists — no tables, no compliance checker |
| Manufacturing rules — **casting / welding / tooling / tolerance (GD&T)** | Only machining, injection, 3D printing, sheet metal |
| **Symbolic planner** | Template planner only — no goal regression, no state-space search, no plan cost |
| **Topology adjacency graph / connectivity reasoning** | Components exist; no adjacency-graph builder for faces/edges |
| **Hybrid reasoning pipeline** (neural→KG→rules→constraints→geometry→mfg→neural) | No orchestration module |
| **Knowledge network** (`knowledge_network/`) | Package does not exist (roadmap P18) |
| Reasoning **evaluation metrics** | `evaluation/reasoning_metrics.py` is a 7-line stub |
| Reasoning **benchmark** | None |

## 3. Duplicated logic

1. AABB predicates 2×: `reasoning/geometry_reasoner.py` vs `world_model/spatial.py`.
2. Numeric constraint solving 2×: `reasoning/constraint_solver.py` vs `cad/parametric/constraints.py`.
3. Report aggregates 4×: `CheckResult` / `CadCheckResult` / `MfgCheck` / `MechanicalResult`.
4. Manufacturing thresholds overlap: `ManufacturingRules` vs `cad/manufacturing/process.py`.
5. Doc/code mismatch: `rule_engine.py` docstring claims it is the substrate for
   `manufacturing_rules`, but manufacturing never imports `RuleEngine`.
6. BFS graph traversal 2×: `KnowledgeGraph` vs `memory/retrieval.graph_search`.

## 4. Integration gaps

- **Bug**: `agents/integration.py:226` `NeuroSymbolicAdapter.reason()` calls
  `.forward(symbolic_facts, neural_state)` on `NeuroSymbolicReasoningEngine`,
  which defines no `forward` → runtime `NotImplementedError`.
- Transformer: only `evaluate_constraints` is wired (`geometry_transformer.py`);
  the returned `symbolic_scores` are discarded; no other reasoning module
  reaches the model.
- World model: `DesignValidator` unused; `WorldModelPlanner.execute()` ops
  `validate`/`simulate`/`check_mechanical` are pass-through no-ops.
- Memory: **zero** reasoning references in `memory/`.
- Confidence / continual learning: zero reasoning references.
- Knowledge network: package missing entirely.
- `KnowledgeGraph` + `SymbolicReasoner` exported but unused in `src/`.

## 5. Architectural improvements

1. Add backward chaining + rule versioning to `RuleEngine` (additive).
2. Add `EngineeringStandards` engine (ISO/ASME/DIN/ANSI/company) with
   compliance checking and a seedable standards knowledge graph.
3. Extend `ManufacturingRules` with casting, welding, tooling, tolerance rules.
4. Add a `SymbolicPlanner` (STRIPS-lite: goal regression, state-space search,
   plan cost) layered over `CADPlan`; keep `TaskPlanner` intact.
5. Add `TopologyReasoner` adjacency-graph + connectivity reasoning.
6. Add a `HybridReasoningPipeline` orchestrating neural → KG → rules →
   constraints → geometry → manufacturing → neural refinement → decision,
   with per-stage reports and pluggable stages.
7. Add constraint dependency propagation + conflict detection + automatic
   repair to `ConstraintSolver` (additive methods).
8. Create `knowledge_network/` (registry of knowledge graphs with global
   search) and wire memory/confidence/continual-learning/agents integrations.
9. Implement `evaluation/reasoning_metrics.py` + `benchmarks/reasoning_benchmarks.py`.
10. Fix the `NeuroSymbolicAdapter` bug and wire the hybrid pipeline into the
    world model facade and the transformer decode hook (additive, flagged).

Everything below is additive; no existing public API is removed or changed.
