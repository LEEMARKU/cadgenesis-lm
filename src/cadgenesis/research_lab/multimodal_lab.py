"""Multimodal Research Lab - New encoders, fusion strategies, representation learning."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any


class ModalityType(str, Enum):
    TEXT = "text"
    CAD = "cad"
    IMAGE = "image"
    SKETCH = "sketch"
    DRAWING = "drawing"
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    PDF = "pdf"


class FusionStrategyType(str, Enum):
    EARLY = "early"
    LATE = "late"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"


@dataclass
class MultimodalExperimentConfig:
    config_id: str
    name: str
    description: str
    modalities: list[ModalityType]
    fusion_strategy: FusionStrategyType
    encoder_configs: dict[str, dict[str, Any]]
    fusion_config: dict[str, Any]
    dataset_config: dict[str, Any]
    evaluation_metrics: list[str]


@dataclass
class MultimodalExperimentResult:
    experiment_id: str
    config: MultimodalExperimentConfig
    metrics: dict[str, float]
    cross_modal_metrics: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    status: str = "completed"
    error: str | None = None


class MultimodalResearchLab:
    def __init__(self):
        self._experiments: dict[str, MultimodalExperimentResult] = {}
        self._configs: dict[str, MultimodalExperimentConfig] = {}
        self._lock = RLock()

    def register_config(self, config: MultimodalExperimentConfig) -> str:
        with self._lock:
            self._configs[config.config_id] = config
            return config.config_id

    def create_encoder_experiment(
        self,
        name: str,
        modality: ModalityType,
        encoder_variants: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        config = MultimodalExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            description=f"Encoder experiment for {modality.value}: {name}",
            modalities=[modality],
            fusion_strategy=FusionStrategyType.LATE,
            encoder_configs={modality.value: {"variants": encoder_variants}},
            fusion_config={},
            dataset_config=kwargs.get("dataset_config", {}),
            evaluation_metrics=kwargs.get("metrics", ["retrieval_accuracy", "generation_quality"]),
        )
        return self.register_config(config)

    def create_fusion_experiment(
        self,
        name: str,
        modalities: list[ModalityType],
        fusion_strategies: list[FusionStrategyType],
        **kwargs,
    ) -> str:
        config = MultimodalExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            description=f"Fusion experiment: {name}",
            modalities=modalities,
            fusion_strategy=fusion_strategies[0]
            if fusion_strategies
            else FusionStrategyType.ATTENTION,
            encoder_configs={m.value: {} for m in modalities},
            fusion_config={"strategies": [s.value for s in fusion_strategies]},
            dataset_config=kwargs.get("dataset_config", {}),
            evaluation_metrics=kwargs.get("metrics", ["cross_modal_accuracy", "alignment_score"]),
        )
        return self.register_config(config)

    def create_representation_experiment(
        self,
        name: str,
        modalities: list[ModalityType],
        representation_dims: list[int],
        **kwargs,
    ) -> str:
        config = MultimodalExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            description=f"Representation learning experiment: {name}",
            modalities=modalities,
            fusion_strategy=FusionStrategyType.HIERARCHICAL,
            encoder_configs={
                m.value: {"representation_dims": representation_dims} for m in modalities
            },
            fusion_config={},
            dataset_config=kwargs.get("dataset_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["linear_probe_accuracy", "clustering_quality"]
            ),
        )
        return self.register_config(config)

    def run_experiment(
        self,
        config_id: str,
        run_fn: Callable[[MultimodalExperimentConfig], dict[str, Any]],
    ) -> MultimodalExperimentResult:
        config = self._configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")

        experiment_id = str(uuid.uuid4())

        try:
            result_data = run_fn(config)
            result = MultimodalExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=result_data.get("metrics", {}),
                cross_modal_metrics=result_data.get("cross_modal_metrics", {}),
                latency_ms=result_data.get("latency_ms", {}),
                status="completed",
            )
        except Exception as e:
            result = MultimodalExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics={},
                cross_modal_metrics={},
                latency_ms={},
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> MultimodalExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[MultimodalExperimentResult]:
        with self._lock:
            return list(self._experiments.values())
