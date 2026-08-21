"""Agent Research Lab - Scheduling, cooperation, communication experiments."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class AgentExperimentType(str, Enum):
    SCHEDULING = "scheduling"
    COOPERATION = "cooperation"
    COMMUNICATION = "communication"
    CONSENSUS = "consensus"
    LOAD_BALANCING = "load_balancing"


@dataclass
class AgentExperimentConfig:
    config_id: str
    name: str
    experiment_type: AgentExperimentType
    description: str
    parameters: dict[str, Any]
    agent_configs: list[dict[str, Any]]
    task_config: dict[str, Any]
    evaluation_metrics: list[str]


@dataclass
class AgentExperimentResult:
    experiment_id: str
    config: AgentExperimentConfig
    metrics: dict[str, float]
    interaction_logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    error: str | None = None


class AgentResearchLab:
    def __init__(self):
        self._experiments: dict[str, AgentExperimentResult] = {}
        self._configs: dict[str, AgentExperimentConfig] = {}
        self._lock = RLock()

    def register_config(self, config: AgentExperimentConfig) -> str:
        with self._lock:
            self._configs[config.config_id] = config
            return config.config_id

    def create_scheduling_experiment(
        self,
        name: str,
        schedulers: list[str],
        task_graphs: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        config = AgentExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=AgentExperimentType.SCHEDULING,
            description=f"Scheduling experiment: {name}",
            parameters={
                "schedulers": schedulers,
                "task_graphs": task_graphs,
            },
            agent_configs=kwargs.get("agent_configs", []),
            task_config=kwargs.get("task_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["makespan", "utilization", "deadline_miss_rate"]
            ),
        )
        return self.register_config(config)

    def create_cooperation_experiment(
        self,
        name: str,
        cooperation_strategies: list[str],
        team_sizes: list[int],
        **kwargs,
    ) -> str:
        config = AgentExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=AgentExperimentType.COOPERATION,
            description=f"Cooperation experiment: {name}",
            parameters={
                "cooperation_strategies": cooperation_strategies,
                "team_sizes": team_sizes,
            },
            agent_configs=kwargs.get("agent_configs", []),
            task_config=kwargs.get("task_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["team_performance", "communication_overhead", "conflict_rate"]
            ),
        )
        return self.register_config(config)

    def create_communication_experiment(
        self,
        name: str,
        protocols: list[str],
        message_types: list[str],
        network_conditions: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        config = AgentExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=AgentExperimentType.COMMUNICATION,
            description=f"Communication experiment: {name}",
            parameters={
                "protocols": protocols,
                "message_types": message_types,
                "network_conditions": network_conditions,
            },
            agent_configs=kwargs.get("agent_configs", []),
            task_config=kwargs.get("task_config", {}),
            evaluation_metrics=kwargs.get("metrics", ["latency", "throughput", "reliability"]),
        )
        return self.register_config(config)

    def run_experiment(
        self,
        config_id: str,
        run_fn: Callable[[AgentExperimentConfig], dict[str, Any]],
    ) -> AgentExperimentResult:
        config = self._configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")

        experiment_id = str(uuid.uuid4())

        try:
            result_data = run_fn(config)
            result = AgentExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=result_data.get("metrics", {}),
                interaction_logs=result_data.get("interaction_logs", []),
                status="completed",
            )
        except Exception as e:
            result = AgentExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics={},
                interaction_logs=[],
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> AgentExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[AgentExperimentResult]:
        with self._lock:
            return list(self._experiments.values())
