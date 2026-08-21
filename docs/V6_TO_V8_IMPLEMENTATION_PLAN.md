# CADGenesis-LM v6.1 → v8.0 Implementation Plan

**Status:** live · **Owner:** build agent + human approval gate
**Baseline:** v6.0 suite 2454 tests (2445 passed / 9 failed — all training-NaN cluster); **v6.1 suite 2477 tests, all pass** (`docs/baseline_v61.txt`); **v6.2 2508** (`docs/baseline_v62.txt`); **v6.3 2536** (`docs/baseline_v63.txt`); **v6.4 2560** (`docs/baseline_v64.txt`); **v6.5 2560** (`docs/baseline_v64.txt`).
**Method:** every milestone is a vertical slice — implement → unit tests → integration tests → docs → config integration → full-suite re-run (recorded baseline per milestone). No placeholder implementations. No removal of working subsystems. Hardware: GTX 1650 4 GB (compute 7.5), CPU-only training runs for mini configs.

---

## 0. Guiding rules (inherited from `docs/v6_roadmap.md`)

1. Never remove working features; upgrades are additive and backward compatible.
2. No fake implementations, no TODOs, no mock components.
3. Everything configurable via `CADConfig` (single source of truth).
4. Tested, documented, integrated, verified before a milestone is declared done.
5. Milestones execute one at a time; after each, the full pytest suite must be green and the baseline recorded before starting the next.
6. Any architectural change is adopted only through the human-approval gate (persisted approval records).

---

## 1. Milestone map

| Milestone | Theme | Modules | Exit criterion |
|-----------|-------|---------|----------------|
| v6.1 | Training stability & inference correctness | `training/`, `transformer/`, `tokenizer/`, `config/`, `inference/` | **DONE** — 2477 tests pass (23 new regression tests) |
| v6.2 | Hardware-aware runtime | `runtime/` (new), `training/trainer.py`, `config/` | **DONE** — 2508 tests pass (31 new tests); presets GTX 1650-4GB / RTX 3050-8GB; autocast fixed; live benchmarks; distillation NaN corner case fixed |
| v6.3 | CAD IR (intermediate representation) | `ir/` (new) | **DONE** — 2536 tests pass (28 new tests); TOON bridge lossless; validator parity with critique + tokenizer gates |
| v6.4 | IR-native execution | `execution/` | **DONE** — `CADExecutionEngine` runs on IR graphs; state/query API; feedback loop via IR diffs |
| v6.5 | Geometry world model | `world_model/`, `geometry/` (new) | **DONE** — spatial predicates (`interference`/`tangent`) added to `SpatialReasoner`; `WorldModelPlanner` provides `plan(goal)` / `execute(plan, graph, material)`; tolerance chain analysis via `SimulationEngine.tolerance` |
| v6.6 | Multimodal grounding | `multimodal/` | CAD/geometry tokens ↔ world-model states; cross-modal attention verified |
| v6.7 | Constraint solver | `reasoning/constraint_solver.py` | Full numerical constraint solving; conflicts detectable & repairable (extend existing) |
| v6.8 | Critics & confidence | `confidence/`, `evaluation/` | Critics score design output; confidence calibration on real generated samples |
| v6.9 | Requirement graph & CAD diff | `reasoning/`, `execution/` | Requirement→feature traceability; structured CAD diffs between revisions |
| v7.0 | Knowledge graph & tool agent | DONE | `reasoning/knowledge_graph`, `agents/` | Tool-using agent over the KG; standards retrieval; verified tool calls |
| v7.1 | Simulation integration | `simulation/` (new) | Lightweight FEA/DFM simulators callable from the IR pipeline |
| v7.2 | Optimization | `optimization/` | Design-parameter optimization (geometry-based, CPU-friendly) |
| v7.3 | Continual learning | `continual_learning/` | Replay + EWC + knowledge anchors wired to trainer |
| v7.4 | Adapter promotion | `adapters/` | LoRA → full-model promotion pipeline with versioning |
| v7.5 | Data factory & adversarial data | `datasets/`, `tokenizer/` | Synthetic data generation with quality scoring; adversarial/edge-case corpus |
| v7.6 | Autonomous benchmark lab & NAS | IN PROGRESS | `benchmarks/`, `transformer/self_designing/` | Reproducible benchmark harness; small-model NAS search |
| v8.0 | Integration, quality gate, docs | repo-wide | Full suite + audit script clean; changelog complete; version 8.0.0 |

---

## 2. Sequencing rationale (dependency chain)

- **v6.2 before v6.3+**: runtime presets bound every later milestone's model-size and memory budgets (4 GB GPU).
- **v6.3 → v6.4 → v6.5 → v6.6**: the IR is the spine; execution consumes it; the world model reasons over executed IR states; multimodal grounds tokens into those states.
- **v6.7, v6.8 parallel-ready**: constraint solver and critics both consume IR; neither depends on the other.
- **v6.9 → v7.0**: requirement graph gives the tool agent its task vocabulary; CAD diff gives it feedback signals.
- **v7.1 → v7.2**: optimization calls the simulators as objective evaluators.
- **v7.3 → v7.4**: adapters are the deployment path for continual-learned updates.
- **v7.5, v7.6**: data factory feeds the benchmark lab; NAS consumes the lab's reproducible metrics.

---

## 3. Cross-cutting constraints

- GPU: GTX 1650 4 GB (compute 7.5) — no full-model training in-repo; use mini/nano configs, CPU training, gradient checkpointing (v6.1 §4.2).
- Runtime device-adaptive budgets (v6.2) replace hardcoded sizes in later milestones.
- Every milestone's new modules are registered in `src/cadgenesis/__init__.py` and mirrored by tests under `tests/`.
- `docs/UPGRADE_STATUS.md` is the single living status file; `docs/CHANGELOG_V6_TO_V8.md` records all changes with evidence.