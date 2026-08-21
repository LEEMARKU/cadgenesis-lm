# CADGenesis-LM v6.1 → v8.0 — Architecture Dependency Graph

**Generated:** 2026-08-19 · modules referenced by their `src/cadgenesis/` paths.
Arrows mean "depends on / consumes". New modules for the upgrade are marked **NEW**.
Everything below the v6.1 line exists and passes tests today.

---

## 1. v6.1 (DONE) — verified subsystem graph

```
config/CADConfig ──────────────► transformer/GeometryAwareTransformer
     │  ▲                            │
     │  └── tokenizer/  ◄───┐        ├─► attention/ (6 backends, safe_softmax, repair)
     │  mini() · vocab      │        ├─► positional/ (RoPE even-dim guard, growable sinusoid)
     │                      │        ├─► ssm/ (trainable forward_cached + dropout)
     └── model params ◄─────┘        ├─► moe/ · interaction/ · embeddings/
                                     │
inference/engine.py ◄─── transformer (greedy/beam/sample/speculative, _mask_bos, GNMT length penalty)
     │  ├── inference/eagle.py (draft head, BOS-masked verification)
     │  └── inference/self_correction.py · mcts.py
training/trainer.py ──► transformer (gradient checkpointing via _maybe_checkpoint; NaN-stable)
```

## 2. v6.2 (NEXT) — HardwareAwareRuntime

```
config/CADConfig ──► runtime/hardware.py  **NEW**  ──► device presets (GTX 1650-4GB / RTX 3050-8GB)
      │                       │
      │                       ├─► runtime/memory_planner.py **NEW** (layer budgets, checkpointing policy)
      │                       ├─► runtime/benchmarks.py **NEW** (live forward/decode measurements)
      │                       └─► training/trainer.py (autocast('cuda') fix, device-aware defaults)
      └─► transformer/ (consumes planned d_model/layers from presets)
```

## 3. v6.3 → v6.4 — CAD IR + IR-native execution

```
tokenizer/ ──► ir/ir_nodes.py **NEW** (node types mirroring TOON grammar)
      │           │
      │           ├─► ir/builder.py **NEW**  (token stream → IR graph)
      │           ├─► ir/serializer.py **NEW** (IR → TOON/JSON round-trip)
      │           └─► ir/validator.py **NEW** (semantic checks, parity with tokenizer validator)
      ▼
execution/execution_engine.py ──► ir/ (executes IR graphs) ──► execution/state.py **NEW**
      └─► execution/feedback.py ──► ir/diff.py **NEW** (v6.9 hook)
```

## 4. v6.5 → v6.6 — Geometry world model + multimodal grounding

```
ir/ ──► world_model/ (extends existing objects/spatial/simulator)
        │
        ├─► geometry/primitives.py **NEW** (parametric box/cyl/extrude/revolve)
        ├─► geometry/predicates.py **NEW** (containment/clearance/intersection)
        └─► geometry/world_state.py **NEW** (step-wise mutation from IR)
              │
              └─► multimodal/ (cross-modal attention: CAD tokens ↔ world states)
```

## 5. v6.7 → v6.8 — Constraint solver + critics/confidence

```
ir/ ──► reasoning/constraint_solver.py (extend: numerical solve, conflict repair)
ir/ ──► confidence/ (critics over generated IR; calibration on real outputs)
         └─► evaluation/ (critic metrics)
```

## 6. v6.9 → v7.0 — Requirement graph + tool agent

```
reasoning/ ──► requirement_graph.py **NEW** (requirements → features traceability)
execution/ ──► cad_diff.py **NEW** (structured revision diffs)
      │
      └─► reasoning/knowledge_graph/ ──► agents/tool_agent.py **NEW** (KG + standards tools,
                                          verified tool-call loop, human-approval gate)
```

## 7. v7.1 → v7.2 — Simulation + optimization

```
ir/ ──► simulation/ **NEW** (lightweight FEA/DFM)
        │
        └─► optimization/ (parameter optimization, simulators as objective)
```

## 8. v7.3 → v7.4 — Continual learning + adapter promotion

```
training/ ──► continual_learning/ (replay + EWC + anchors wired to trainer)
      │
      └─► adapters/ (LoRA → full-model promotion, versioned)
```

## 9. v7.5 → v7.6 — Data factory + benchmark lab / NAS

```
datasets/ ──► data_factory.py **NEW** (quality-scored synthetic data)
      │
      └─► benchmarks/ (harness) ──► transformer/self_designing/ (small-model NAS)
```

## 10. v8.0 — Integration & quality gate

```
all of the above ──► scripts/audit_repo.py · full pytest suite · docs/CHANGELOG_V6_TO_V8.md
```

---

## 11. Key dependency rules

1. **ir/ is the spine** — no milestone past v6.4 may bypass it for CAD semantics.
2. **runtime/ is consulted first** — any milestone choosing model sizes or batch sizes must read the v6.2 preset for the active device.
3. **Nothing replaces a working subsystem** — v6.7 extends `constraint_solver.py`, v6.5 extends `world_model/`; existing tests must keep passing.
4. **Human-approval gate** — v7.0 tool agent and v7.6 NAS adopt changes only with persisted approval records.