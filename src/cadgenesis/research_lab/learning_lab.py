"""Learning Research Lab - Distillation, continual learning, adapter experiments."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class LearningExperimentType(str, Enum):
    DISTILLATION = "distillation"
    CONTINUAL_LEARNING = "continual_learning"
    ADAPTER = "adapter"
    PEFT = "peft"
    SELF_SUPERVISED = "self_supervised"
    RLHF = "rlhf"
    RLAIF = "rlaif"


@dataclass
class LearningExperimentConfig:
    config_id: str
    name: str
    experiment_type: LearningExperimentType
    description: str
    parameters: dict[str, Any]
    model_config: dict[str, Any]
    data_config: dict[str, Any]
    evaluation_metrics: list[str]


@dataclass
class LearningExperimentResult:
    experiment_id: str
    config: LearningExperimentConfig
    metrics: dict[str, float]
    training_curves: dict[str, list[float]] = field(default_factory=dict)
    forgetting_metrics: dict[str, float] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None


class LearningResearchLab:
    def __init__(self):
        self._experiments: dict[str, LearningExperimentResult] = {}
        self._configs: dict[str, LearningExperimentConfig] = {}
        self._lock = RLock()

    def register_config(self, config: LearningExperimentConfig) -> str:
        with self._lock:
            self._configs[config.config_id] = config
            return config.config_id

    def create_distillation_experiment(
        self,
        name: str,
        teacher_models: list[str],
        student_architectures: list[str],
        distillation_methods: list[str],
        **kwargs,
    ) -> str:
        config = LearningExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=LearningExperimentType.DISTILLATION,
            description=f"Distillation experiment: {name}",
            parameters={
                "teacher_models": teacher_models,
                "student_architectures": student_architectures,
                "distillation_methods": distillation_methods,
            },
            model_config=kwargs.get("model_config", {}),
            data_config=kwargs.get("data_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["student_accuracy", "compression_ratio", "latency_reduction"]
            ),
        )
        return self.register_config(config)

    def create_continual_learning_experiment(
        self,
        name: str,
        methods: list[str],
        task_sequences: list[list[str]],
        **kwargs,
    ) -> str:
        config = LearningExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=LearningExperimentType.CONTINUAL_LEARNING,
            description=f"Continual learning experiment: {name}",
            parameters={
                "methods": methods,
                "task_sequences": task_sequences,
            },
            model_config=kwargs.get("model_config", {}),
            data_config=kwargs.get("data_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["avg_accuracy", "forgetting", "forward_transfer", "backward_transfer"]
            ),
        )
        return self.register_config(config)

    def create_adapter_experiment(
        self,
        name: str,
        adapter_types: list[str],
        ranks: list[int],
        base_models: list[str],
        **kwargs,
    ) -> str:
        config = LearningExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=LearningExperimentType.ADAPTER,
            description=f"Adapter experiment: {name}",
            parameters={
                "adapter_types": adapter_types,
                "ranks": ranks,
                "base_models": base_models,
            },
            model_config=kwargs.get("model_config", {}),
            data_config=kwargs.get("data_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["task_accuracy", "parameter_efficiency", "training_time"]
            ),
        )
        return self.register_config(config)

    def run_experiment(
        self,
        config_id: str,
        run_fn: Callable[[LearningExperimentConfig], dict[str, Any]],
    ) -> LearningExperimentResult:
        config = self._configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")

        experiment_id = str(uuid.uuid4())

        try:
            result_data = run_fn(config)
            result = LearningExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=result_data.get("metrics", {}),
                training_curves=result_data.get("training_curves", {}),
                forgetting_metrics=result_data.get("forgetting_metrics", {}),
                status="completed",
            )
        except Exception as e:
            result = LearningExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics={},
                training_curves={},
                forgetting_metrics={},
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> LearningExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[LearningExperimentResult]:
        with self._lock:
            return list(self._experiments.values())
