# CADGenesis-LM Enterprise Architecture

This document describes the enterprise repository layout that CADGenesis-LM
follows. It is the canonical reference for where code, data, tooling, and
deployment artifacts live.

## Repository layout

```
CADGenesis-LM/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── SECURITY.md
├── CITATION.cff
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── Makefile
├── .gitignore
├── .env.example
├── docker-compose.yml
│
├── configs/          # experiment / deployment configuration files
├── docs/             # architecture & engineering documentation
├── scripts/          # CI and operational scripts
├── tools/            # development utilities
├── src/              # source code (src layout)
├── tests/            # pytest suite (mirrors src/cadgenesis)
├── benchmarks/       # performance benchmarks
├── deployments/      # deployment manifests and configs
├── docker/           # Dockerfiles
├── notebooks/        # analysis / experiment notebooks
├── examples/         # runnable examples and demos
├── plugins/          # optional plugins
├── sdk/              # standalone SDK modules (e.g. TOON serialization)
└── external/         # external references; large data lives outside the repo
```

## Source code

All application code lives under `src/cadgenesis/` (a src layout) and is
installed as an editable package:

```
src/cadgenesis/
│
├── tokenizer/            # Autonomous CAD tokenizer (vocabulary, evolution, TOON backend)
├── transformer/          # Geometry-Aware & Self-Designing Transformer
├── memory/               # Layer-integrated memory (8 semantic stores + 9th long-term,
│                         #   neural pools, router, retrieval, compression, persistence v2,
│                         #   semantic→neural bridge, transformer augmentation)
├── agents/               # Multi-agent platform (18-agent fleet, event bus, DAG scheduling, pipeline, orchestrator)
├── reasoning/            # Neuro-symbolic reasoning (rules + backward chaining, constraints,
│                         #   standards library, symbolic planner, hybrid pipeline, KG) (M19)
├── knowledge_network/    # Multi-source engineering knowledge network (KG + standards) (M19)
├── execution/            # CAD execution engine (FreeCAD/OCC backends)
├── continual_learning/   # Rehearsal, EWC, adapter isolation
├── adapters/             # PEFT / LoRA / QLoRA / self-evolving adapter bank
├── confidence/           # Confidence, uncertainty, calibration, risk
├── distillation/         # Multi-teacher distillation & RLAIF
├── datasets/             # Dataset loaders and pipelines
├── training/             # Trainer, distributed, FSDP/DeepSpeed, checkpoints
├── inference/            # Inference engine (greedy / beam decoding)
├── serving/              # HTTP/gRPC serving layer
├── optimization/         # Quantization, pruning, ONNX, kernels
├── evaluation/           # Metrics, benchmark runner, report generation
├── multimodal/           # Multimodal encoders, cross-modal, fusion, embedding space (M19)
├── world_model/          # World-model reasoning: spatial, mechanical, simulator, planning (M19)
├── telemetry/            # Tracing and metrics collection
├── monitoring/           # Health checks, drift, alerts
├── utils/                # Shared helpers
├── logging/              # Logging configuration
├── config/               # CADConfig & sub-configurations (single source of truth)
└── cli/                  # Command-line entrypoints
```

## External data (outside the git repository)

Large artifacts are **not** stored in the source repository. They live in an
external directory `CADGenesis-Data/` (commonly at the same level as the repo
or configured via `CADGENESIS_DATA_ROOT`):

```
CADGenesis-Data/
├── datasets/
├── checkpoints/
├── pretrained_models/
├── adapters/
├── logs/
├── cache/
├── experiments/
├── tensorboard/
├── generated_models/
├── cad_outputs/
├── simulations/
└── backups/
```

This directory can grow beyond 100 GB without affecting the source repository
size. It is git-ignored via `.gitignore` (`external/CADGenesis-Data/`).

## Import conventions

- Package imports use the top-level package: `from cadgenesis.config import CADConfig`.
- Module paths follow the directory layout, e.g. `cadgenesis.transformer.geometry_transformer`.
- The SDK (`sdk/`) is importable as `sdk.toon`, `sdk.toon_extended`.
- Entrypoints: `python -m cadgenesis.train` (training), `scripts/train.py` (forwarding wrapper).
