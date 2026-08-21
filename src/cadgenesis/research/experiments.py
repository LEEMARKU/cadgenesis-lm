"""
cadgenesis.research.experiments
===============================
Experiment tracking for CADGenesis-LM research infrastructure.

- Unique experiment IDs (``exp_<timestamp>_<nonce>``) with human labels
- Metadata, hyperparameters, metrics, artifacts and notes
- File-backed (JSON) with atomic writes and optional SQLite mirror
- Hyperparameter tracking is first-class: optimizer, scheduler, lr,
  adapters, architecture are captured as a structured ``Hyperparams``
- Live metric logging: ``log_metric(name, value, step)``

Filesystem layout::

    <root>/<experiment_id>/
        meta.json       # experiment record (id, params, metrics, notes)
        artifacts/      # artifact registry copies
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.research.experiments")

META_FILE = "meta.json"


def new_experiment_id() -> str:
    return f"exp_{int(time.time())}_{uuid.uuid4().hex[:8]}"


@dataclass
class Hyperparams:
    """Structured hyperparameter snapshot for tracking & reproducibility."""

    optimizer: str = "adamw"
    scheduler: str = "cosine"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 1
    warmup_steps: int = 0
    adapter: str = "none"  # none | lora | qlora | ...
    architecture: str = "geometry_aware_transformer"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "extra": dict(self.extra)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Hyperparams:
        known = {k: v for k, v in data.items() if k != "extra"}
        return cls(**known, extra=dict(data.get("extra", {})))

    def fingerprint(self) -> str:
        from cadgenesis.utils.hashing import content_hash

        return content_hash(self.to_dict())


@dataclass
class ExperimentRecord:
    """One tracked experiment."""

    id: str
    name: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "running"  # running | completed | failed | aborted
    hyperparams: Hyperparams = field(default_factory=Hyperparams)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    best_metric: float | None = None
    best_metric_name: str = "loss"

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        entry = {"value": value}
        if step is not None:
            entry["step"] = step
        self.metrics.setdefault(name, []).append(entry)

    def update_best(self, metric: str = "loss", minimize: bool = True) -> None:
        values = [
            m.get("value")
            for m in self.metrics.get(metric, [])
            if isinstance(m.get("value"), (int, float))
        ]
        if not values:
            return
        best = min(values) if minimize else max(values)
        self.best_metric = float(best)
        self.best_metric_name = metric

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "hyperparams": self.hyperparams.to_dict(),
            "metrics": {k: list(v) for k, v in self.metrics.items()},
            "metadata": dict(self.metadata),
            "notes": list(self.notes),
            "artifacts": list(self.artifacts),
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentRecord:
        record = cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            created_at=float(data.get("created_at", 0.0)),
            status=str(data.get("status", "completed")),
            hyperparams=Hyperparams.from_dict(data.get("hyperparams", {})),
            metadata=dict(data.get("metadata", {})),
            notes=list(data.get("notes", [])),
            artifacts=list(data.get("artifacts", [])),
            best_metric=data.get("best_metric"),
            best_metric_name=str(data.get("best_metric_name", "loss")),
        )
        record.metrics = {k: list(v) for k, v in data.get("metrics", {}).items()}
        return record


class ExperimentTracker:
    """File-backed experiment tracker with atomic persistence."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: dict[str, ExperimentRecord] = {}
        self._load_all()

    def _load_all(self) -> None:
        for directory in self.root.iterdir():
            meta = directory / META_FILE
            if meta.exists():
                try:
                    record = ExperimentRecord.from_dict(
                        json.loads(meta.read_text(encoding="utf-8"))
                    )
                    self._records[record.id] = record
                except (ValueError, KeyError) as exc:
                    logger.warning("skipping unreadable experiment %s: %s", directory, exc)

    def create(
        self,
        name: str = "",
        hyperparams: Hyperparams | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        if hyperparams is not None and not isinstance(hyperparams, Hyperparams):
            hyperparams = Hyperparams.from_dict(hyperparams)
        record = ExperimentRecord(
            id=experiment_id or new_experiment_id(),
            name=name,
            hyperparams=hyperparams or Hyperparams(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._records[record.id] = record
            self._persist(record)
        logger.info("experiment created: %s", record.id)
        return record

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            return self._records.get(experiment_id)

    def all(self, status: str | None = None) -> list[ExperimentRecord]:
        with self._lock:
            records = list(self._records.values())
        if status:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.created_at)

    def log_metric(
        self, experiment_id: str, name: str, value: float, step: int | None = None
    ) -> None:
        with self._lock:
            record = self._require(experiment_id)
            record.log_metric(name, value, step)
            self._persist(record)

    def add_note(self, experiment_id: str, note: str) -> None:
        with self._lock:
            record = self._require(experiment_id)
            record.notes.append(note)
            self._persist(record)

    def set_status(
        self, experiment_id: str, status: str, metric: str = "loss", minimize: bool = True
    ) -> None:
        with self._lock:
            record = self._require(experiment_id)
            record.status = status
            record.update_best(metric, minimize)
            self._persist(record)

    def attach_artifact(
        self, experiment_id: str, path: str | os.PathLike[str], copy: bool = False
    ) -> str:
        """Record (and optionally copy) an artifact for an experiment."""
        source = Path(path)
        with self._lock:
            record = self._require(experiment_id)
            artifact_dir = self.root / record.id / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if copy and source.exists():
                target = artifact_dir / source.name
                target.write_bytes(source.read_bytes())
                record.artifacts.append(str(target))
            else:
                record.artifacts.append(str(source))
            self._persist(record)
            return record.artifacts[-1]

    def best(
        self, metric: str = "loss", minimize: bool = True, limit: int = 1
    ) -> list[ExperimentRecord]:
        """Experiments ranked by ``metric`` (None-valued entries ranked last)."""
        candidates = [r for r in self.all() if r.metrics.get(metric)]
        ranked = sorted(
            candidates,
            key=lambda r: min(m.get("value", float("inf")) for m in r.metrics[metric]),
            reverse=not minimize,
        )
        return ranked[:limit]

    def _require(self, experiment_id: str) -> ExperimentRecord:
        record = self._records.get(experiment_id)
        if record is None:
            raise KeyError(f"unknown experiment {experiment_id!r}")
        return record

    def _persist(self, record: ExperimentRecord) -> None:
        directory = self.root / record.id
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".meta-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record.to_dict(), handle, indent=2)
            os.replace(tmp, directory / META_FILE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def export_summary(self) -> dict[str, Any]:
        return {
            "count": len(self._records),
            "running": sum(1 for r in self._records.values() if r.status == "running"),
            "completed": sum(1 for r in self._records.values() if r.status == "completed"),
            "failed": sum(1 for r in self._records.values() if r.status == "failed"),
            "experiments": [r.to_dict() for r in self.all()],
        }


__all__ = ["ExperimentRecord", "ExperimentTracker", "Hyperparams", "new_experiment_id"]
