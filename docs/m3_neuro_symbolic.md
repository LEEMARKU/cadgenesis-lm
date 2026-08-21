# M3 — Neuro-Symbolic Reasoning

Milestone M3 of the CADGenesis-LM v6.0 Ultimate Architecture roadmap
(`docs/v6_roadmap.md`) completes the Neuro-Symbolic Reasoning pillar
(`cadgenesis.reasoning`): every stub is replaced by a tested, documented,
production-quality module.

## Scope

The nine reasoning modules cover the full symbolic verification stack used to
make CAD designs sound before execution: declarative rules, numeric constraint
satisfaction, geometric reasoning, manufacturing (DFM) heuristics, workflow
planning, a safe symbolic expression evaluator, topological analysis, an
engineering knowledge graph, and the validation orchestrator that ties it all
together.

## Modules delivered

| Module | Contents |
| --- | --- |
| `rule_engine.py` | `Rule` / `RuleResult` / `RuleEngine` / `make_rule`; severity levels, priority ordering, forward chaining (`run`), `violations`, `summary` |
| `constraint_solver.py` | `Variable` (bounded), `Constraint` (linear, `==`/`<=`/`>=`), projection-based `ConstraintSolver.solve` -> `Solution`, `check_consistency` |
| `geometry_reasoner.py` | `Primitive` (7 kinds), analytical volumes, AABB, `overlaps`, `clearance`, `contains`/`check_fit`, `combined_bounds`, validation |
| `topology.py` | `TopologyStats` / `TopologyAnalyzer`; Euler characteristic, genus, manifold/closed checks, connected components, Euler-Poincare consistency, `analyze_mesh` |
| `knowledge_graph.py` | `GraphNode` / `GraphEdge` / `KnowledgeGraph`; typed weighted edges, neighbors/predecessors, BFS shortest path, related-expansion, JSON persistence |
| `manufacturing_rules.py` | `ManufacturingRules` / `MfgCheck` / `ManufacturingAssessment`; machining, injection molding, 3D printing, sheet-metal DFM checks with tunable thresholds |
| `planner.py` | `PlanningStep` / `CADPlan` / `TaskPlanner`; dependency ordering, cycle detection, critical path, workflow templates, rule-based refinement |
| `symbolic_reasoner.py` | `SymbolicExpression` (AST whitelist — no arbitrary code), `SymbolicReasoner.check_constraint` / `check_implication` / `check_token_consistency`, `VerificationResult` |
| `validator.py` | `CheckResult` / `ValidationReport` / `DesignValidator`; orchestrates rule + constraint + geometry + manufacturing + topology + custom checks |
| `__init__.py` | Package facade exporting the full public API (27 names) |

## Design notes

- **Pure Python.** Every M3 module is dependency-free (no torch), so the
  reasoning layer runs anywhere and tests are instant. The existing torch-based
  `neuro_symbolic.py` (`NeuroSymbolicReasoningEngine`) is preserved unchanged.
- **Shared substrate.** `validator.py` composes the other modules instead of
  re-implementing logic; `manufacturing_rules.py` produces `MfgCheck` objects
  the validator consumes directly.
- **Safe evaluation.** `SymbolicExpression` parses with `ast` and evaluates only
  a whitelist of operators/functions/constants; anything else raises.
- **Correct semantics.** The constraint solver's projection converges to a
  feasible assignment and reports a diagnostic when bounds conflict; topology
  derives genus from the Euler-Poincare relation and flags inconsistent counts.

## Verification

```text
pytest           668 passed (143 new reasoning tests)
ruff check       clean for cadgenesis.reasoning and tests/reasoning
audit_repo.py    178 modules · 278 public APIs · 14 812 LOC · 82 stubs
                 Neuro-Symbolic Reasoning pillar: OK
```
