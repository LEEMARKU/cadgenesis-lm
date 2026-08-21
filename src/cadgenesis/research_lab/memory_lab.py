"""Memory Research Lab - Memory routing, compression, retrieval experiments."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any


class MemoryExperimentType(str, Enum):
    ROUTING = "routing"
    COMPRESSION = "compression"
    RETRIEVAL = "retrieval"
    CONSOLIDATION = "consolidation"
    PERSISTENCE = "persistence"


@dataclass
class MemoryExperimentConfig:
    """Configuration for a memory experiment."""

    config_id: str
    name: str
    experiment_type: MemoryExperimentType
    description: str
    parameters: dict[str, Any]
    dataset_config: dict[str, Any]
    evaluation_metrics: list[str]


@dataclass
class MemoryExperimentResult:
    """Results from a memory experiment."""

    experiment_id: str
    config: MemoryExperimentConfig
    metrics: dict[str, float]
    latency_ms: dict[str, float]
    memory_usage_mb: float
    compression_ratio: float | None = None
    status: str = "completed"
    error: str | None = None


class MemoryResearchLab:
    """Lab for experimenting with memory system variants."""

    def __init__(self):
        self._experiments: dict[str, MemoryExperimentResult] = {}
        self._configs: dict[str, MemoryExperimentConfig] = {}
        self._lock = RLock()

    def register_config(self, config: MemoryExperimentConfig) -> str:
        with self._lock:
            self._configs[config.config_id] = config
            return config.config_id

    def create_routing_experiment(
        self,
        name: str,
        routing_strategies: list[str],
        context_types: list[str],
        **kwargs,
    ) -> str:
        config = MemoryExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=MemoryExperimentType.ROUTING,
            description=f"Routing experiment: {name}",
            parameters={
                "routing_strategies": routing_strategies,
                "context_types": context_types,
            },
            dataset_config=kwargs.get("dataset_config", {}),
            evaluation_metrics=kwargs.get("metrics", ["accuracy", "latency", "recall@k"]),
        )
        return self.register_config(config)

    def create_compression_experiment(
        self,
        name: str,
        compression_methods: list[str],
        target_ratios: list[float],
        **kwargs,
    ) -> str:
        config = MemoryExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=MemoryExperimentType.COMPRESSION,
            description=f"Compression experiment: {name}",
            parameters={
                "compression_methods": compression_methods,
                "target_ratios": target_ratios,
            },
            dataset_config=kwargs.get("dataset_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["compression_ratio", "reconstruction_loss", "retrieval_quality"]
            ),
        )
        return self.register_config(config)

    def create_retrieval_experiment(
        self,
        name: str,
        retrieval_methods: list[str],
        top_k_values: list[int],
        **kwargs,
    ) -> str:
        config = MemoryExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=MemoryExperimentType.RETRIEVAL,
            description=f"Retrieval experiment: {name}",
            parameters={
                "retrieval_methods": retrieval_methods,
                "top_k_values": top_k_values,
            },
            dataset_config=kwargs.get("dataset_config", {}),
            evaluation_metrics=kwargs.get("metrics", ["precision@k", "recall@k", "mrr", "ndcg"]),
        )
        return self.register_config(config)

    def run_experiment(
        self,
        config_id: str,
        run_fn: Callable[[MemoryExperimentConfig], dict[str, Any]],
    ) -> MemoryExperimentResult:
        config = self._configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")

        experiment_id = str(uuid.uuid4())

        try:
            result_data = run_fn(config)
            result = MemoryExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=result_data.get("metrics", {}),
                latency_ms=result_data.get("latency_ms", {}),
                memory_usage_mb=result_data.get("memory_usage_mb", 0),
                compression_ratio=result_data.get("compression_ratio"),
                status="completed",
            )
        except Exception as e:
            result = MemoryExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics={},
                latency_ms={},
                memory_usage_mb=0,
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> MemoryExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(
        self, experiment_type: MemoryExperimentType | None = None
    ) -> list[MemoryExperimentResult]:
        with self._lock:
            results = list(self._experiments.values())
            if experiment_type:
                results = [r for r in results if r.config.experiment_type == experiment_type]
            return results
