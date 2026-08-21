"""Experiment Registry - Configurations, checkpoints, metrics, reports."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PROMOTED = "promoted"


@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    lab_type: str  # transformer, memory, multimodal, world_model, agent, neuro_symbolic, learning
    config: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentCheckpoint:
    checkpoint_id: str
    experiment_id: str
    step: int
    metrics: dict[str, float]
    model_state: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str
    status: ExperimentStatus
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    duration_seconds: float | None = None


class ExperimentRegistry:
    """Central registry for all research experiments."""

    def __init__(self, storage_path: str = "./research_experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._experiments: dict[str, ExperimentConfig] = {}
        self._results: dict[str, ExperimentResult] = {}
        self._checkpoints: dict[str, list[ExperimentCheckpoint]] = {}
        self._lock = RLock()

    def create_experiment(self, config: ExperimentConfig) -> str:
        with self._lock:
            self._experiments[config.experiment_id] = config
            self._results[config.experiment_id] = ExperimentResult(
                experiment_id=config.experiment_id,
                status=ExperimentStatus.CREATED,
            )
            self._checkpoints[config.experiment_id] = []
            self._save_experiment(config)
            return config.experiment_id

    def get_experiment(self, experiment_id: str) -> ExperimentConfig | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> bool:
        with self._lock:
            result = self._results.get(experiment_id)
            if not result:
                return False
            result.status = status
            if status == ExperimentStatus.RUNNING:
                result.started_at = time.time()
            elif status in (
                ExperimentStatus.COMPLETED,
                ExperimentStatus.FAILED,
                ExperimentStatus.CANCELLED,
            ):
                result.completed_at = time.time()
                if result.started_at:
                    result.duration_seconds = result.completed_at - result.started_at
            self._save_result(result)
            return True

    def record_metrics(self, experiment_id: str, metrics: dict[str, float]) -> bool:
        with self._lock:
            result = self._results.get(experiment_id)
            if not result:
                return False
            result.metrics.update(metrics)
            self._save_result(result)
            return True

    def add_checkpoint(self, checkpoint: ExperimentCheckpoint) -> bool:
        with self._lock:
            if checkpoint.experiment_id not in self._checkpoints:
                return False
            self._checkpoints[checkpoint.experiment_id].append(checkpoint)
            self._save_checkpoint(checkpoint)
            return True

    def add_artifact(self, experiment_id: str, name: str, path: str) -> bool:
        with self._lock:
            result = self._results.get(experiment_id)
            if not result:
                return False
            result.artifacts[name] = path
            self._save_result(result)
            return True

    def add_log(self, experiment_id: str, log: str) -> bool:
        with self._lock:
            result = self._results.get(experiment_id)
            if not result:
                return False
            result.logs.append(f"[{time.time()}] {log}")
            self._save_result(result)
            return True

    def get_result(self, experiment_id: str) -> ExperimentResult | None:
        with self._lock:
            return self._results.get(experiment_id)

    def get_checkpoints(self, experiment_id: str) -> list[ExperimentCheckpoint]:
        with self._lock:
            return list(self._checkpoints.get(experiment_id, []))

    def list_experiments(
        self,
        lab_type: str | None = None,
        status: ExperimentStatus | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ExperimentConfig]:
        with self._lock:
            experiments = list(self._experiments.values())
            if lab_type:
                experiments = [e for e in experiments if e.lab_type == lab_type]
            if status:
                experiments = [
                    e for e in experiments if self._results[e.experiment_id].status == status
                ]
            if tags:
                experiments = [e for e in experiments if any(t in e.tags for t in tags)]
            experiments.sort(key=lambda e: e.created_at, reverse=True)
            return experiments[:limit]

    def _save_experiment(self, config: ExperimentConfig) -> None:
        path = self.storage_path / f"{config.experiment_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(config), f, indent=2)

    def _save_result(self, result: ExperimentResult) -> None:
        path = self.storage_path / f"{result.experiment_id}_result.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

    def _save_checkpoint(self, checkpoint: ExperimentCheckpoint) -> None:
        path = self.storage_path / f"{checkpoint.experiment_id}_checkpoints.json"
        checkpoints_data = [asdict(cp) for cp in self._checkpoints[checkpoint.experiment_id]]
        with open(path, "w") as f:
            json.dump(checkpoints_data, f, indent=2, default=str)

    def load_from_storage(self) -> int:
        """Load all experiments from storage."""
        count = 0
        for path in self.storage_path.glob("*.json"):
            if path.name.endswith("_result.json") or path.name.endswith("_checkpoints.json"):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                config = ExperimentConfig(**data)
                self._experiments[config.experiment_id] = config
                self._results[config.experiment_id] = ExperimentResult(
                    experiment_id=config.experiment_id,
                    status=ExperimentStatus.CREATED,
                )
                self._checkpoints[config.experiment_id] = []
                count += 1
            except Exception:
                pass
        return count
