"""World Model Research Lab - Planning, simulation, latent representations experiments."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class WorldModelExperimentType(str, Enum):
    PLANNING = "planning"
    SIMULATION = "simulation"
    LATENT_REPRESENTATION = "latent_representation"
    SPATIAL_REASONING = "spatial_reasoning"
    MECHANICAL_REASONING = "mechanical_reasoning"
    ASSEMBLY_REASONING = "assembly_reasoning"
    FUNCTIONAL_REASONING = "functional_reasoning"


@dataclass
class WorldModelExperimentConfig:
    config_id: str
    name: str
    experiment_type: WorldModelExperimentType
    description: str
    parameters: dict[str, Any]
    scenario_config: dict[str, Any]
    evaluation_metrics: list[str]


@dataclass
class WorldModelExperimentResult:
    experiment_id: str
    config: WorldModelExperimentConfig
    metrics: dict[str, float]
    trajectory_data: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None


class WorldModelResearchLab:
    def __init__(self):
        self._experiments: dict[str, WorldModelExperimentResult] = {}
        self._configs: dict[str, WorldModelExperimentConfig] = {}
        self._lock = RLock()

    def register_config(self, config: WorldModelExperimentConfig) -> str:
        with self._lock:
            self._configs[config.config_id] = config
            return config.config_id

    def create_planning_experiment(
        self,
        name: str,
        planning_algorithms: list[str],
        scenario_types: list[str],
        horizon: int = 10,
        **kwargs,
    ) -> str:
        config = WorldModelExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=WorldModelExperimentType.PLANNING,
            description=f"Planning experiment: {name}",
            parameters={
                "planning_algorithms": planning_algorithms,
                "scenario_types": scenario_types,
                "horizon": horizon,
            },
            scenario_config=kwargs.get("scenario_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["success_rate", "plan_quality", "planning_time"]
            ),
        )
        return self.register_config(config)

    def create_simulation_experiment(
        self,
        name: str,
        simulators: list[str],
        physics_engines: list[str],
        **kwargs,
    ) -> str:
        config = WorldModelExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=WorldModelExperimentType.SIMULATION,
            description=f"Simulation experiment: {name}",
            parameters={
                "simulators": simulators,
                "physics_engines": physics_engines,
            },
            scenario_config=kwargs.get("scenario_config", {}),
            evaluation_metrics=kwargs.get("metrics", ["accuracy", "speed", "stability"]),
        )
        return self.register_config(config)

    def create_latent_representation_experiment(
        self,
        name: str,
        encoder_types: list[str],
        latent_dims: list[int],
        **kwargs,
    ) -> str:
        config = WorldModelExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=WorldModelExperimentType.LATENT_REPRESENTATION,
            description=f"Latent representation experiment: {name}",
            parameters={
                "encoder_types": encoder_types,
                "latent_dims": latent_dims,
            },
            scenario_config=kwargs.get("scenario_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["reconstruction_error", "downstream_task_performance"]
            ),
        )
        return self.register_config(config)

    def run_experiment(
        self,
        config_id: str,
        run_fn: Callable[[WorldModelExperimentConfig], dict[str, Any]],
    ) -> WorldModelExperimentResult:
        config = self._configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")

        experiment_id = str(uuid.uuid4())

        try:
            result_data = run_fn(config)
            result = WorldModelExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=result_data.get("metrics", {}),
                trajectory_data=result_data.get("trajectory_data", {}),
                status="completed",
            )
        except Exception as e:
            result = WorldModelExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics={},
                trajectory_data={},
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> WorldModelExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[WorldModelExperimentResult]:
        with self._lock:
            return list(self._experiments.values())
