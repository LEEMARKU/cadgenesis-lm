"""Experimental Transformer Lab - New attention mechanisms, FFN architectures, routing."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, cast

import torch
import torch.nn as nn

from cadgenesis.config import CADConfig


class AttentionMechanismType(str, Enum):
    """Types of attention mechanisms to experiment with."""

    STANDARD = "standard"
    FLASH = "flash"
    SPARSE = "sparse"
    LINEAR = "linear"
    PERFORMER = "performer"
    NYSTRÖM = "nystrom"
    LONGFORMER = "longformer"
    BIGBIRD = "bigbird"
    CUSTOM = "custom"


class FFNType(str, Enum):
    """Types of feed-forward networks."""

    STANDARD = "standard"
    GLU = "glu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    MOE = "moe"
    CUSTOM = "custom"


@dataclass
class ArchitectureSpec:
    """Specification for a transformer architecture variant."""

    spec_id: str
    name: str
    description: str
    attention_type: AttentionMechanismType
    ffn_type: FFNType
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    ff_dim: int
    dropout: float = 0.1
    attention_config: dict[str, Any] = field(default_factory=dict)
    ffn_config: dict[str, Any] = field(default_factory=dict)
    custom_modules: dict[str, Callable] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "name": self.name,
            "description": self.description,
            "attention_type": self.attention_type.value,
            "ffn_type": self.ffn_type.value,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "attention_config": self.attention_config,
            "ffn_config": self.ffn_config,
        }


@dataclass
class ExperimentResult:
    """Results from an architecture experiment."""

    experiment_id: str
    spec: ArchitectureSpec
    metrics: dict[str, float]
    training_time: float
    memory_peak_mb: float
    flops: float
    parameters: int
    status: str  # completed, failed, running
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


class ExperimentalTransformerLab:
    """Lab for experimenting with transformer architecture variants."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._experiments: dict[str, ExperimentResult] = {}
        self._specs: dict[str, ArchitectureSpec] = {}
        self._lock = RLock()

    def register_spec(self, spec: ArchitectureSpec) -> str:
        """Register a new architecture specification."""
        with self._lock:
            self._specs[spec.spec_id] = spec
            return spec.spec_id

    def create_spec(
        self,
        name: str,
        description: str,
        attention_type: AttentionMechanismType = AttentionMechanismType.STANDARD,
        ffn_type: FFNType = FFNType.STANDARD,
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ff_dim: int = 3072,
        **kwargs,
    ) -> str:
        """Create and register a new architecture spec."""
        spec = ArchitectureSpec(
            spec_id=str(uuid.uuid4()),
            name=name,
            description=description,
            attention_type=attention_type,
            ffn_type=ffn_type,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            ff_dim=ff_dim,
            **kwargs,
        )
        return self.register_spec(spec)

    def build_model(self, spec: ArchitectureSpec) -> nn.Module:
        """Build a model from an architecture spec."""
        # This is a simplified builder - in production would use actual components
        from cadgenesis.transformer import GeometryAwareTransformer

        config = {
            "hidden_dim": spec.hidden_dim,
            "num_layers": spec.num_layers,
            "num_heads": spec.num_heads,
            "ff_dim": spec.ff_dim,
            "dropout": spec.dropout,
        }
        return GeometryAwareTransformer(cast(CADConfig, config)).to(self.device)

    def run_experiment(
        self,
        spec_id: str,
        train_fn: Callable[[nn.Module], dict[str, float]],
        eval_fn: Callable[[nn.Module], dict[str, float]] | None = None,
    ) -> ExperimentResult:
        """Run an experiment with the given spec."""
        spec = self._specs.get(spec_id)
        if not spec:
            raise ValueError(f"Spec {spec_id} not found")

        experiment_id = str(uuid.uuid4())
        model = self.build_model(spec)

        try:
            start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None

            if start_time:
                start_time.record()

            metrics = train_fn(model)

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                assert start_time is not None
                training_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                import time

                training_time = time.time() - start_time  # type: ignore

            memory_peak = (
                torch.cuda.max_memory_allocated() / 1024 / 1024 if self.device == "cuda" else 0
            )

            # Calculate FLOPs and parameters
            params = sum(p.numel() for p in model.parameters())
            flops = 0  # Would use fvcore or similar

            result = ExperimentResult(
                experiment_id=experiment_id,
                spec=spec,
                metrics=metrics,
                training_time=training_time,
                memory_peak_mb=memory_peak,
                flops=flops,
                parameters=params,
                status="completed",
            )
        except Exception as e:
            result = ExperimentResult(
                experiment_id=experiment_id,
                spec=spec,
                metrics={},
                training_time=0,
                memory_peak_mb=0,
                flops=0,
                parameters=0,
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> ExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[ExperimentResult]:
        with self._lock:
            return list(self._experiments.values())

    def compare_architectures(self, experiment_ids: list[str]) -> dict[str, Any]:
        """Compare multiple architecture experiments."""
        fetched = [self.get_experiment(eid) for eid in experiment_ids]
        results: list[ExperimentResult] = []
        for r in fetched:
            if r is not None and r.status == "completed":
                results.append(r)

        if not results:
            return {"error": "No completed experiments to compare"}

        metrics_keys: set[str] = set()
        for r in results:
            metrics_keys.update(r.metrics.keys())

        comparison = {
            "experiments": [r.experiment_id for r in results],
            "specs": [r.spec.to_dict() for r in results],
            "metrics": {k: [r.metrics.get(k, 0) for r in results] for k in metrics_keys},
            "training_times": [r.training_time for r in results],
            "memory_peaks": [r.memory_peak_mb for r in results],
            "parameters": [r.parameters for r in results],
        }

        return comparison
