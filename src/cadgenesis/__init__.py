"""
CADGenesis-LM v6.1
==================
A Self-Evolving Neuro-Symbolic Geometry Foundation Model for Generative
Parametric CAD Design — Ultimate Architecture.

Package structure:
    cadgenesis/
        config/                 — CADConfig dataclass (single source of truth)
        runtime/                — HardwareAwareRuntime (device presets, memory planning, benchmarks)
        tokenizer/              — Autonomous CAD Tokenizer
        transformer/            — Geometry-Aware / Self-Designing Transformer
        memory/                 — Layer-Integrated Memory Pools
        agents/                 — Multi-Agent Internal Transformer
        reasoning/              — Neuro-Symbolic Reasoning Engine
        execution/              — CAD Execution Intelligence Engine
        adapters/               — LoRA / QLoRA / Adapter Bank
        continual_learning/     — Continual Learning subsystem
        distillation/           — Multi-Teacher Distillation Engine
        alignment/              — RLAIF + Constitutional AI
        confidence/             — Confidence-Aware Intelligence
        datasets/               — Dataset loaders and pipelines
        training/               — Training infrastructure
        inference/              — Inference engine
        serving/                — Model serving layer
        optimization/           — Inference/model optimizations
        evaluation/             — Metrics and evaluation harnesses
        telemetry/              — Tracing and metrics collection
        monitoring/             — Health checks and observability
        utils/                  — Shared helpers
        logging/                — Logging configuration
        cli/                    — Command-line entrypoints
"""

__version__ = "8.0.0"
__author__ = "CADGenesis Team"

from cadgenesis.pillar_overview import get_pillar_overview

__all__ = ["__author__", "__version__", "get_pillar_overview"]
