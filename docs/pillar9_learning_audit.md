# Pillar 9 — Learning System: Repository Audit

Audit performed before implementation (v6.0 roadmap, Pillar 9).

## 1. Implemented training components

| Module | Status | Notes |
|---|---|---|
| `training/trainer.py` | **Implemented** | `CADTrainer` (AdamW, pad-ignored CE, grad accumulation, fp16/bf16 autocast + GradScaler, cosine-with-warmup LambdaLR, `save_checkpoint`/`load_checkpoint`, `train_epoch`/`validate`). Also `MultiModalCADDataset`, `cad_collate_fn`. Hard torch import. |
| `adapters/lora.py` | **Implemented** | Real LoRA math `W0 + (α/r)·B·A`: `LoRALinear`, `apply_lora`. Missing merge/unmerge, save/load, param filtering. Hard torch import. |
| `adapters/manager.py` | **Implemented** | `SelfEvolvingAdapterBank` + `AdapterMetadata` (register / evaluate_and_promote / trigger_rollback). Pure Python. |
| `continual_learning/replay_buffer.py` | **Implemented** | `ReplayBuffer` + `ReplaySample`: importance-weighted sampling, semantic recall via memory. Fixed capacity, caller-supplied importance (not adaptive). Pure Python. |
| `distillation/distillation_engine.py` | **Partial** | `MultiTeacherDistillationEngine` (single-teacher KD loss: hard CE + T²·KL), `TeacherConsensusEngine` (trivial "first sequence + match fraction"). Not actually multi-teacher. |
| `distillation/distill_pipeline.py` | **Partial** | `TeacherModelInterface` (rule-based fallback; API branch empty), `QualityFilteringEngine` (real, wired to CAD execution + safety), `AutomatedDatasetGenPipeline` (5 hardcoded prompts), `DistillationLossPipeline`, `SelfImprovementLoop` (rule-simulated). |
| `alignment/constitutional_ai.py` | **Partial** | `CADConstitutionalPrinciples` (4 real rules), `RLAIFRewardModel` (scalar reward head), `SafetyInterventionEngine` (block/warn/suggest/allow). No self-critique, no iterative refinement, no preference optimization. |
| `datasets/multimodal.py` | **Implemented** | `MultimodalDataset`, `MultimodalBatch`, `MultimodalBatchCollator`, `MultimodalSample`. Hard torch import. |
| Transformer confidence machinery | **Implemented** | `ConfidenceHead`, `ConfidenceLoss`, `CADSequenceLoss`, `UncertaintyAttention`, `EarlyExitGate` (used by Pillar 10). |

## 2. Stubs (docstring-only) — 24 files

**`training/` (9):** `distributed.py`, `checkpoint.py`, `deepspeed.py`, `fsdp.py`, `scheduler.py`, `optimizer.py`, `callbacks.py`, `metrics.py`, `profiler.py`.

**`adapters/` (7):** `qlora.py`, `peft.py`, `router.py`, `lifecycle.py`, `versioning.py`, `rollback.py`, `promotion.py`.

**`distillation/` (9):** `soft_labels.py`, `hard_labels.py`, `consensus.py`, `critique.py`, `pipeline.py`, `rlaif.py`, `synthetic.py`, `teachers/openai_teacher.py`, `teachers/open_source_teacher.py`.

**`continual_learning/` (6):** `ewc.py`, `continual_trainer.py`, `adapter_isolation.py`, `evaluator.py`, `updater.py`, `knowledge_anchor.py`.

**`optimization/` (4):** `quantization.py`, `kernels.py`, `pruning.py`, `onnx.py` (+ empty `__init__.py`).

## 3. Missing learning capabilities (vs. mission)

- Curriculum / mixed datasets, dataset versioning — absent.
- Self-supervised learning (masked modeling, contrastive, next-operation, representation) — absent.
- Multi-teacher / co-distillation / cross-architecture / progressive distillation — absent (name only).
- Synthetic generation of assemblies, simulations, manufacturing cases, edge cases, failure cases — absent (`synthetic.py` stub; pipeline covers single-part only).
- Constitutional self-critique + iterative refinement — absent.
- RLAIF preference optimization (Bradley-Terry / DPO) — absent (`rlaif.py` stub).
- EWC (Fisher) — absent; adaptive replay — absent; online/incremental learning — absent; forgetting evaluation — absent.
- PEFT (adapter/prefix/prompt tuning, IA3) — absent.
- QLoRA / 4-bit training — absent.
- Quantization (FP16/BF16/INT8/INT4/GPTQ/AWQ) — absent (`optimization/quantization.py` stub).
- Gradient checkpointing — config flag `gradient_checkpointing` exists but nothing consumes it.
- TurboVec / embedding acceleration / embedding caching — absent entirely.
- Distributed multi-GPU/multi-node, automatic checkpointing/resume (beyond plain save/load), LR scheduler factory, callbacks, metrics, profiler — absent.
- Autonomous self-learning (self-reflection, error analysis, failure detection, automatic retraining, data-quality evaluation) — absent (`SelfImprovementLoop` is a simulation).
- Backward-compatible torch guards: modules hard-import torch (`training/`, `adapters/`, `distillation/`, `alignment/`, `datasets/` fail to import without torch).

## 4. Duplicated functionality

1. KD loss: `distillation_engine.compute_loss` vs `distill_pipeline.DistillationLossPipeline.compute_loss` (thin wrapper, same math).
2. Confidence blending: `confidence/confidence_engine.py` entropy+head blend vs transformer `ConfidenceHead` sigmoid — duplicated ad-hoc blending.
3. Scheduler: inline `LambdaLR` inside `CADTrainer.configure_scheduler` vs `training/scheduler.py` stub (should become the single factory).
4. Adapter promotion logic: `SelfEvolvingAdapterBank.evaluate_and_promote` thresholds vs `adapters/promotion.py` stub.

## 5. Integration gaps

- `agents/integration.py::ContinualLearningHooks` calls `replay_buffer.record` and `ewc.consolidate` — **neither exists** (hooks silently return `None`).
- `training/__init__.py` exports only 3 names; new trainer features (callbacks, schedulers) unexported.
- `CADConfig.gradient_checkpointing` flag is dead.
- CLI `cli/train.py` uses raw `CADTrainer`; no automatic checkpointing/resume helpers, no dataset versioning.
- Distillation pipeline never persists dataset versions; memory persistence for learning artifacts is ad-hoc.
- No benchmark suite for training convergence / forgetting / adapter quality (Pillar 9 Step 6).

## 6. Architecture plan (backward compatible)

1. **`training/`** — fill 9 stubs (scheduler factory, optimizer factory, callbacks, metrics, checkpoint w/ auto + resume, distributed (DDP + multi-node), profiler, DeepSpeed/FSDP plugin interfaces); add `datasets.py` (curriculum + mixed + registry w/ versioning), `self_supervised.py` (MLM/contrastive/next-op/representation), `gradient_checkpoint.py`, `self_improvement.py` (autonomous learning). Extend `trainer.py` additively where needed.
2. **`adapters/`** — fill 7 stubs (PEFT: adapter/prefix/prompt/IA3; router for automatic selection; lifecycle/versioning/rollback/promotion); extend `lora.py` (merge/unmerge, save/load); `qlora.py` (4-bit quantized adapters).
3. **`optimization/`** — quantization (FP16/BF16/INT8/INT4/GPTQ/AWQ), kernels (TurboVec acceleration + embedding cache), pruning, ONNX export.
4. **`distillation/`** — real multi-teacher/co/cross-arch/progressive distillation; soft/hard label providers; consensus; critique; pipeline; RLAIF (preference data + BT loss + DPO); synthetic generators (programs/prompts/assemblies/simulations/mfg/edge/failure).
5. **`continual_learning/`** — EWC (Fisher), continual trainer, adapter isolation, forgetting evaluator, online updater, knowledge anchors, adaptive replay (extend `ReplayBuffer`).
6. **`alignment/`** — self-critique + iterative refinement over an expanded constitution (CAD safety + manufacturing rules).
7. Integrations: fix `ContinualLearningHooks` targets, wire dataset versioning + memory persistence, confidence integration.
8. Tests under `tests/{training,adapters,optimization,continual_learning,distillation,alignment}/`, benchmarks, docs.

Nothing existing is removed; all stub files are replaced in place; implemented modules are extended additively.
