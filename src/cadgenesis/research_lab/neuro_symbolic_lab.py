"""Neuro-Symbolic Research Lab - Rule learning, symbolic planning, hybrid reasoning."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class NeuroSymbolicExperimentType(str, Enum):
    RULE_LEARNING = "rule_learning"
    SYMBOLIC_PLANNING = "symbolic_planning"
    HYBRID_REASONING = "hybrid_reasoning"
    CONSTRAINT_REASONING = "constraint_reasoning"
    KNOWLEDGE_GRAPH = "knowledge_graph"


@dataclass
class NeuroSymbolicExperimentConfig:
    config_id: str
    name: str
    experiment_type: NeuroSymbolicExperimentType
    description: str
    parameters: dict[str, Any]
    knowledge_base_config: dict[str, Any]
    evaluation_metrics: list[str]


@dataclass
class NeuroSymbolicExperimentResult:
    experiment_id: str
    config: NeuroSymbolicExperimentConfig
    metrics: dict[str, float]
    learned_rules: list[dict[str, Any]] = field(default_factory=list)
    proofs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    error: str | None = None


class NeuroSymbolicResearchLab:
    def __init__(self):
        self._experiments: dict[str, NeuroSymbolicExperimentResult] = {}
        self._configs: dict[str, NeuroSymbolicExperimentConfig] = {}
        self._lock = RLock()

    def register_config(self, config: NeuroSymbolicExperimentConfig) -> str:
        with self._lock:
            self._configs[config.config_id] = config
            return config.config_id

    def create_rule_learning_experiment(
        self,
        name: str,
        learning_algorithms: list[str],
        rule_templates: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        config = NeuroSymbolicExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=NeuroSymbolicExperimentType.RULE_LEARNING,
            description=f"Rule learning experiment: {name}",
            parameters={
                "learning_algorithms": learning_algorithms,
                "rule_templates": rule_templates,
            },
            knowledge_base_config=kwargs.get("kb_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["rule_accuracy", "coverage", "precision", "recall"]
            ),
        )
        return self.register_config(config)

    def create_symbolic_planning_experiment(
        self,
        name: str,
        planners: list[str],
        domain_complexity: list[str],
        **kwargs,
    ) -> str:
        config = NeuroSymbolicExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=NeuroSymbolicExperimentType.SYMBOLIC_PLANNING,
            description=f"Symbolic planning experiment: {name}",
            parameters={
                "planners": planners,
                "domain_complexity": domain_complexity,
            },
            knowledge_base_config=kwargs.get("kb_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["plan_success_rate", "plan_length", "planning_time"]
            ),
        )
        return self.register_config(config)

    def create_hybrid_reasoning_experiment(
        self,
        name: str,
        neural_components: list[str],
        symbolic_components: list[str],
        integration_strategies: list[str],
        **kwargs,
    ) -> str:
        config = NeuroSymbolicExperimentConfig(
            config_id=str(uuid.uuid4()),
            name=name,
            experiment_type=NeuroSymbolicExperimentType.HYBRID_REASONING,
            description=f"Hybrid reasoning experiment: {name}",
            parameters={
                "neural_components": neural_components,
                "symbolic_components": symbolic_components,
                "integration_strategies": integration_strategies,
            },
            knowledge_base_config=kwargs.get("kb_config", {}),
            evaluation_metrics=kwargs.get(
                "metrics", ["reasoning_accuracy", "explanation_quality", "consistency"]
            ),
        )
        return self.register_config(config)

    def run_experiment(
        self,
        config_id: str,
        run_fn: Callable[[NeuroSymbolicExperimentConfig], dict[str, Any]],
    ) -> NeuroSymbolicExperimentResult:
        config = self._configs.get(config_id)
        if not config:
            raise ValueError(f"Config {config_id} not found")

        experiment_id = str(uuid.uuid4())

        try:
            result_data = run_fn(config)
            result = NeuroSymbolicExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=result_data.get("metrics", {}),
                learned_rules=result_data.get("learned_rules", []),
                proofs=result_data.get("proofs", []),
                status="completed",
            )
        except Exception as e:
            result = NeuroSymbolicExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics={},
                learned_rules=[],
                proofs=[],
                status="failed",
                error=str(e),
            )

        with self._lock:
            self._experiments[experiment_id] = result

        return result

    def get_experiment(self, experiment_id: str) -> NeuroSymbolicExperimentResult | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[NeuroSymbolicExperimentResult]:
        with self._lock:
            return list(self._experiments.values())
