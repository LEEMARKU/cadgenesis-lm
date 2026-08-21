# CADGenesis-LM v6.0

CADGenesis-LM is a synthetic CAD language model for generative parametric CAD
design — a Self-Evolving Neuro-Symbolic Geometry Foundation Model (Ultimate
Architecture v6.0).

The repository follows the **CADGenesis-LM Enterprise Architecture** described
in `docs/architecture.md`: all source lives under `src/cadgenesis/`, with the
SDK under `sdk/`, tests under `tests/`, and large external data kept outside
the repo in `external/CADGenesis-Data/`.

## Layout

```
src/cadgenesis/        source code (tokenizer, transformer, memory, agents, ...)
tests/                 pytest suite
benchmarks/            performance benchmarks
sdk/                   standalone SDK modules (TOON serialization)
configs/  docs/        configuration and documentation
scripts/  tools/       scripts and utilities
deployments/  docker/  deployment artifacts
notebooks/  examples/  notebooks and runnable examples
external/              external data references (data stored out-of-repo)
```

## Quick Start

Install in editable mode:

```bash
python -m pip install -e ".[dev,bpe]"
```

Train the mini verification model:

```bash
python -m cadgenesis.train
```

Use custom training settings:

```bash
python -m cadgenesis.train --epochs 10 --batch-size 32 --train-size 1000 --valid-size 200 --output-dir outputs/cadgenesis_train
```

Train the full CADGenesis-LM v2.0 architecture instead of the mini verification model:

```bash
python -m cadgenesis.train --model-size full --epochs 5 --batch-size 8 --train-size 200 --valid-size 50 --output-dir outputs/cadgenesis_train_full
```

Resume training from a saved checkpoint:

```bash
python -m cadgenesis.train --resume-from outputs/cadgenesis_train/best_checkpoint.pt
```

The root-level script `scripts/train.py` forwards to the package training
entrypoint.

## Validation

Run the test suite with:

```bash
python -m pytest -q
```

Audit the repository against the v6.0 acceptance criteria:

```bash
python scripts/audit_repo.py            # human-readable report
python scripts/audit_repo.py --strict   # fail on any remaining stub / missing test
python scripts/audit_repo.py --json     # machine-readable report
```

Run benchmarks:

```bash
python benchmarks/attention_benchmarks.py
python benchmarks/tokenizer_benchmarks.py
```

## Implemented Subsystems (v6.0)

The full transformation plan is tracked in `docs/v6_roadmap.md`; the automated
repository audit lives at `scripts/audit_repo.py` and is run after every
milestone.

### Foundations (M1) — `cadgenesis.utils`, `.logging`, `.telemetry`, `.monitoring`
Production helpers and observability: thread-safe metrics registry
(counter/gauge/histogram), hierarchical tracing spans, structured logging with
console/file/JSON sinks, health checks, PSI/KL/JS drift detection, and an
alerting framework.  All knobs are configurable via
`CADConfig.observability` (`ObservabilityConfig`).

### Foundation Model & Tokenizer completeness (M2) — `cadgenesis.transformer`, `.tokenizer`
The Foundation Model pillar is now fully self-contained:

- **Transformer** — reusable `TokenEmbedding`/`TypeEmbedding`/`CombinedInputEmbedding`
  input embeddings; `EncoderStack` / `DecoderStack` over the existing
  `CADTransformerBlock` with layer-gate / head-weight / memory-refine hooks and a
  guaranteed confidence output; `LMHead` (weight tying) + `ConfidenceHead`;
  `MaskedCrossEntropyLoss` / `ConfidenceLoss` / `CADSequenceLoss`; standalone
  `ExpertRouter` with load-balancing loss.  Facade (`transformer.py`) and shim
  modules keep every historical import path working.
- **Tokenizer** — canonical token-definition modules per family, a generated
  numeric table and an aggregate `cad_tokens.py` registry (1 296 CAD tokens +
  23 specials), plus standalone lossless `compression.py`, `serialization.py`
  and `validation.py` helpers and facade modules (`tokenizer`, `token_evolution`,
  `vocabulary_manager`).

### Neuro-Symbolic Reasoning (M3) — `cadgenesis.reasoning`
The Neuro-Symbolic Reasoning pillar is complete.  Every previously stubbed
reasoning module is now implemented, tested, and dependency-free:

- **Declarative rules** — `RuleEngine` with forward chaining, severities,
  priorities, `violations()` and `summary()`.
- **Constraint solving** — bounded `Variable`s and linear `Constraint`s solved by
  projection; `ConstraintSolver.solve()` returns a feasible `Solution` or a
  diagnostic when bounds conflict.
- **Geometry & topology** — analytical primitive volumes/AABBs/overlap/clearance/
  fit checks (`GeometryReasoner`) and Euler characteristic, genus, manifold and
  closed-surface analysis (`TopologyAnalyzer`).
- **Manufacturing (DFM)** — `ManufacturingRules` for machining, injection
  molding, 3D printing and sheet metal with tunable thresholds.
- **Planning** — `TaskPlanner` with dependency ordering, cycle detection,
  critical path and workflow templates.
- **Symbolic & knowledge** — safe AST-whitelist `SymbolicExpression` evaluation
  and `KnowledgeGraph` for engineering relationships.
- **Orchestration** — `DesignValidator` composes every check category into one
  `ValidationReport`; `reasoning/__init__.py` exports the full public API.

### Memory System completeness (M4) — `cadgenesis.memory`
The Layer-Integrated Memory System now has a full **semantic memory layer**
complementing the torch neural bank (`LayerIntegratedMemorySystem`):

- **Eight domain pools** — `WorkingMemory`, `SessionMemory`, `UserMemory`,
  `ProjectMemory`, `CADMemory`, `EngineeringMemory`, `ManufacturingMemory`,
  `SimulationMemory` — each a bounded, scored store built on the shared
  `MemoryStore` / `MemoryEntry` foundation.
- **Cross-pool retrieval** — `MemoryRetrieval` merges ranked, deduplicated
  top-k hits across pools; `MemoryRouter` routes a query to the most relevant
  pool first.
- **Lifecycle** — `MemoryPruner` enforces capacity / staleness / importance
  eviction policies; `MemoryPersistence` saves and restores every pool
  atomically as JSON.
- **Facade** — `MemorySystem` composes all eight pools + retriever + router +
  pruner + persistence and wires pool capacities from `MemoryConfig`.

### Multi-Agent Intelligence (M5) — `cadgenesis.agents`
An external orchestration layer complements the torch in-model agent bus
(`MultiAgentSystem`, preserved unchanged):

- **Protocol & registry** — `Agent` ABC with `AgentRequest` / `AgentResult`,
  role/action validation, and `AgentCoordinator` dispatch across a role-keyed
  registry.
- **Collaboration substrate** — `MessageBus` (topic pub/sub with bounded
  history and subscriber fault isolation), `SharedMemory` (thread-safe
  blackboard), `TaskScheduler` (priority + dependency ordering, cycle
  detection), `ConsensusEngine` (majority / weighted / mean / unanimity).
- **Eight role agents** — `PlannerAgent`, `GeometryAgent`, `ConstraintAgent`,
  `ManufacturingAgent`, `OptimizationAgent`, `AssemblyAgent`,
  `SimulationAgent`, `ValidationAgent` — wrapping the M3 reasoning APIs and
  ready to be driven standalone or through the coordinator.

### Self-Designing Transformer — `cadgenesis.transformer.self_designing`
The `SelfDesigningTransformer` wraps the existing `GeometryAwareTransformer`
backbone (nothing is rebuilt) and adds:

- **Neural Architecture Search** (`NeuralArchitectureSearch`, `ArchitectureSpec`,
  `ArchitectureEvaluator`) — random and evolutionary (µ+λ) search over depth,
  width, head layout and MoE switch.
- **Dynamic Layer Routing** (`DynamicLayerRouter`) — per-token Gumbel-Sigmoid
  layer skip gates.
- **Adaptive Attention Heads** (`AdaptiveAttentionHeadSelector`) — per-token
  head gating.
- **Sparse Expert Growth** (`cadgenesis.transformer.moe.SparseMoEFFN`) —
  growable top-k experts with load-balancing auxiliary loss.
- **Reversible Layer Pruning** (`LayerPruningController`) — gradient-free
  importance pruning, exact-skip, fully reversible.
- **Automatic Rollback** (`AutomaticRollback`) — versioned weight snapshots
  with metric-based automatic restore.

```python
from cadgenesis.transformer.self_designing import SelfDesigningTransformer

model = SelfDesigningTransformer(config)  # forward-compatible API
model.grow_experts(1)  # add experts to every MoE block
model.prune_layers(0.25)  # reversibly prune weakest layers
best, score, summary = model.search_architecture(dataset)  # NAS
model.snapshot(metric)
model.check_performance(metric)  # auto-rollback
```

### Layer-Integrated Memory Pools — `cadgenesis.memory.memory_pools`
Every encoder/decoder block reads the combined memory bank and writes back a
differentiable refinement into the working pool.  The 8 required pools
(working, session, project, user, cad, engineering, manufacturing, simulation)
provide 288 slots; `retrieve()` performs cross-pool retrieval and `refine()`
per-layer write-back.

### Autonomous CAD Tokenizer — `cadgenesis.tokenizer`
The native tokenization system (TOON remains the serialization backend):

- **Dynamic vocabulary growth** (`CADVocabulary.register`, `remove_token`,
  `merge_tokens`, `split_token`, `trim_unused`).
- **Vocabulary evolution** (`cadgenesis.tokenizer.evolution`) — analyzes a
  corpus, registers frequent unknowns and merges frequent pairs:
  `tokenizer.evolve(sequences)`.
- **TOON backend adapter** (`cadgenesis.tokenizer.toon_backend.ToonBackend`) —
  serialize sequences and whole vocabularies (including slot layouts) to TOON
  text while TOON itself stays untouched:
  `tokenizer.serialize_to_toon(seq)`, `tokenizer.deserialize_from_toon(text)`.
- On-the-fly auto-registration via `encode_cad_token(tok, auto_register=True)`.

### Inference Engine — `cadgenesis.inference`
```python
from cadgenesis.inference import CADInferenceEngine

engine = CADInferenceEngine(model, tokenizer)
result = engine.greedy("create a steel box 50mm wide", max_len=64)  # or engine.beam(...)
result.tokens  # generated CAD token strings
result.confidence  # mean model confidence
result.toon  # TOON-serialized CAD sequence
```

### Engineering Trust Infrastructure (Pillar 13) — `cadgenesis.trust`
Cryptographic provenance, integrity verification, and tamper-resistant audit trails for datasets, models, CAD assets, experiments, plugins, and adapters. Blockchain backend is optional and pluggable (local, Ethereum, Hyperledger, Polygon).

### Frontier AI Research Laboratory (Pillar 16) — `cadgenesis.research_lab`
Isolated research environment for developing, evaluating, and benchmarking new AI ideas without affecting the production model. Includes labs for transformer, memory, multimodal, world model, agents, neuro-symbolic, and learning research. Safe promotion pipeline: Experimental → Benchmark → Validation → Regression Tests → Human Approval → Production.

### Autonomous AI Research Laboratory (Pillar 17) — `cadgenesis.autonomous_research`
AI Research Laboratory capable of automatically designing, executing, evaluating, and documenting machine learning experiments while keeping humans in control of approvals. Research Planner → Hypothesis Generator → Experiment Planner → Automated Runner → Benchmark Evaluator → Statistical Analyzer → Hyperparameter Search → Architecture Comparator → Failure Analyzer → Report Generator → Human Approval Pipeline.

### Global Engineering Knowledge Network (Pillar 18) — `cadgenesis.knowledge_network`
Unified engineering knowledge platform with knowledge graph, standards library (ISO, ASME, ANSI, DIN), material database, manufacturing knowledge base, formula library, CAD component library, patent knowledge base, and hybrid retrieval engine (vector + graph + symbolic + BM25).

### Industrial Digital Twin (Pillar 19) — `cadgenesis.digital_twin`
Comprehensive Industrial Digital Twin Platform synchronizing CAD models, simulations, manufacturing systems, operational data, IoT sensors, robotics, and predictive analytics. Product, factory, machine, and process digital twins with real-time bidirectional synchronization.

### Autonomous Engineering Platform (Pillar 20) — `cadgenesis.autonomous_platform`
Final pillar integrating all 20 pillars into one unified engineering intelligence platform. 22-stage workflow: Multimodal Understanding → Intent Extraction → Requirement Graph → World Model → Knowledge/Memory Retrieval → Planner Agent → Task Graph → Multi-Agent Collaboration → Neuro-Symbolic Reasoning → CAD Generation → Geometry/Constraint Validation → Simulation → Manufacturing Analysis → Optimization → Reliability Verification → Documentation Generation → Digital Twin Validation → Human Review → Final Engineering Package.

### SDK — `sdk/`
The TOON serialization format (`sdk.toon`) and its schema-aware/streaming extensions (`sdk.toon_extended`) ship as a standalone SDK.

## External Data

Large data (datasets, checkpoints, logs, generated models) is stored outside
the repository. See `external/README.md` and `docs/architecture.md`.
