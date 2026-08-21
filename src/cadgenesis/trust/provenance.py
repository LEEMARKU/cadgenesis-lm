"""Provenance tracking for datasets, models, CAD assets, and experiments."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProvenanceEventType(str, Enum):
    """Types of provenance events."""

    CREATED = "created"
    MODIFIED = "modified"
    DERIVED = "derived"
    AUGMENTED = "augmented"
    PREPROCESSED = "preprocessed"
    TRAINED = "trained"
    FINE_TUNED = "fine_tuned"
    DISTILLED = "distilled"
    QUANTIZED = "quantized"
    EXPORTED = "exported"
    IMPORTED = "imported"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


@dataclass
class ProvenanceEvent:
    """A single event in the provenance chain."""

    event_id: str
    event_type: ProvenanceEventType
    timestamp: float
    actor: str  # user, system, or agent identifier
    description: str
    input_hashes: list[str] = field(default_factory=list)
    output_hashes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "description": self.description,
            "input_hashes": self.input_hashes,
            "output_hashes": self.output_hashes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceEvent:
        return cls(
            event_id=data["event_id"],
            event_type=ProvenanceEventType(data["event_type"]),
            timestamp=data["timestamp"],
            actor=data["actor"],
            description=data["description"],
            input_hashes=data.get("input_hashes", []),
            output_hashes=data.get("output_hashes", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProvenanceBase:
    """Base class for all provenance records."""

    provenance_id: str
    name: str
    version: str
    content_hash: str
    created_at: float
    created_by: str
    events: list[ProvenanceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    owner: str | None = None
    license: str | None = None
    tags: list[str] = field(default_factory=list)

    def add_event(
        self,
        event_type: ProvenanceEventType,
        actor: str,
        description: str,
        input_hashes: list[str] | None = None,
        output_hashes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceEvent:
        event = ProvenanceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            actor=actor,
            description=description,
            input_hashes=input_hashes or [],
            output_hashes=output_hashes or [],
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def get_history(self) -> list[ProvenanceEvent]:
        return list(self.events)

    def verify_integrity(self, current_content: bytes) -> bool:
        current_hash = hashlib.sha256(current_content).hexdigest()
        return current_hash == self.content_hash


@dataclass
class DatasetProvenance(ProvenanceBase):
    """Provenance tracking for datasets."""

    source: str = ""  # original source (URL, path, generator)
    format: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    size_bytes: int = 0
    preprocessing_steps: list[dict[str, Any]] = field(default_factory=list)
    augmentation_config: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, float] = field(default_factory=dict)  # train/val/test ratios

    def record_preprocessing(
        self, step: str, config: dict[str, Any], actor: str
    ) -> ProvenanceEvent:
        self.preprocessing_steps.append({"step": step, "config": config, "timestamp": time.time()})
        return self.add_event(
            ProvenanceEventType.PREPROCESSED,
            actor,
            f"Preprocessing step: {step}",
            metadata={"step": step, "config": config},
        )

    def record_augmentation(self, config: dict[str, Any], actor: str) -> ProvenanceEvent:
        self.augmentation_config = config
        return self.add_event(
            ProvenanceEventType.AUGMENTED,
            actor,
            "Data augmentation applied",
            metadata={"config": config},
        )

    def record_split(self, splits: dict[str, float], actor: str) -> ProvenanceEvent:
        self.splits = splits
        return self.add_event(
            ProvenanceEventType.MODIFIED,
            actor,
            f"Dataset split: {splits}",
            metadata={"splits": splits},
        )


@dataclass
class ModelProvenance(ProvenanceBase):
    """Provenance tracking for models and adapters."""

    architecture: str = ""
    framework: str = ""  # pytorch, tensorflow, onnx, etc.
    base_model: str | None = None  # provenance_id of base model
    training_config: dict[str, Any] = field(default_factory=dict)
    training_data_hash: str | None = None
    checkpoint_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    adapter_type: str | None = None  # lora, qlora, peft, prefix, prompt, ia3
    adapter_config: dict[str, Any] = field(default_factory=dict)
    teacher_models: list[str] = field(default_factory=list)  # for distillation
    quantization: str | None = None  # fp16, bf16, int8, int4, gptq, awq

    def record_training(
        self,
        config: dict[str, Any],
        data_hash: str,
        metrics: dict[str, float],
        actor: str,
    ) -> ProvenanceEvent:
        self.training_config = config
        self.training_data_hash = data_hash
        self.metrics = metrics
        return self.add_event(
            ProvenanceEventType.TRAINED,
            actor,
            "Model training completed",
            metadata={"config": config, "metrics": metrics},
        )

    def record_fine_tuning(
        self,
        base_model_id: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        actor: str,
    ) -> ProvenanceEvent:
        self.base_model = base_model_id
        self.training_config = config
        self.metrics = metrics
        return self.add_event(
            ProvenanceEventType.FINE_TUNED,
            actor,
            "Fine-tuning completed",
            metadata={"base_model": base_model_id, "config": config, "metrics": metrics},
        )

    def record_distillation(
        self,
        teacher_ids: list[str],
        config: dict[str, Any],
        metrics: dict[str, float],
        actor: str,
    ) -> ProvenanceEvent:
        self.teacher_models = teacher_ids
        self.training_config = config
        self.metrics = metrics
        return self.add_event(
            ProvenanceEventType.DISTILLED,
            actor,
            "Knowledge distillation completed",
            metadata={"teachers": teacher_ids, "config": config, "metrics": metrics},
        )

    def record_quantization(
        self, method: str, config: dict[str, Any], actor: str
    ) -> ProvenanceEvent:
        self.quantization = method
        return self.add_event(
            ProvenanceEventType.QUANTIZED,
            actor,
            f"Model quantized to {method}",
            metadata={"method": method, "config": config},
        )


@dataclass
class CADAssetProvenance(ProvenanceBase):
    """Provenance tracking for CAD assets."""

    cad_format: str = ""  # step, iges, stl, obj, etc.
    geometry_hash: str = ""
    topology_hash: str = ""
    design_intent: str | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)
    features: list[dict[str, Any]] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    manufacturing_info: dict[str, Any] = field(default_factory=dict)
    assembly_structure: dict[str, Any] | None = None
    revision: int = 1
    parent_revision: int | None = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    exports: list[dict[str, Any]] = field(default_factory=list)

    def record_revision(
        self, revision: int, parent_revision: int, actor: str, description: str
    ) -> ProvenanceEvent:
        self.revision = revision
        self.parent_revision = parent_revision
        return self.add_event(
            ProvenanceEventType.MODIFIED,
            actor,
            f"Revision {revision}: {description}",
            metadata={"revision": revision, "parent": parent_revision},
        )

    def record_approval(
        self, approver: str, role: str, status: str, comments: str = ""
    ) -> ProvenanceEvent:
        approval = {
            "approver": approver,
            "role": role,
            "status": status,
            "comments": comments,
            "timestamp": time.time(),
        }
        self.approvals.append(approval)
        return self.add_event(
            ProvenanceEventType.APPROVED if status == "approved" else ProvenanceEventType.REJECTED,
            approver,
            f"Design {status} by {approver} ({role})",
            metadata=approval,
        )

    def record_export(
        self, format: str, path: str, actor: str, metadata: dict[str, Any] | None = None
    ) -> ProvenanceEvent:
        export_info = {
            "format": format,
            "path": path,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.exports.append(export_info)
        return self.add_event(
            ProvenanceEventType.EXPORTED,
            actor,
            f"Exported to {format}: {path}",
            metadata=export_info,
        )


@dataclass
class ExperimentProvenance(ProvenanceBase):
    """Provenance tracking for experiments."""

    hypothesis: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    dataset_hashes: list[str] = field(default_factory=list)
    model_hashes: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    status: str = "planned"  # planned, running, completed, failed, cancelled
    started_at: float | None = None
    completed_at: float | None = None

    def start(self, actor: str) -> ProvenanceEvent:
        self.status = "running"
        self.started_at = time.time()
        return self.add_event(
            ProvenanceEventType.CREATED,
            actor,
            "Experiment started",
            metadata={"configuration": self.configuration},
        )

    def complete(
        self,
        results: dict[str, Any],
        artifacts: list[dict[str, Any]],
        actor: str,
    ) -> ProvenanceEvent:
        self.status = "completed"
        self.completed_at = time.time()
        self.results = results
        self.artifacts = artifacts
        return self.add_event(
            ProvenanceEventType.VALIDATED,
            actor,
            "Experiment completed",
            metadata={"results": results, "artifacts": artifacts},
        )

    def fail(self, error: str, actor: str) -> ProvenanceEvent:
        self.status = "failed"
        self.completed_at = time.time()
        return self.add_event(
            ProvenanceEventType.REJECTED,
            actor,
            f"Experiment failed: {error}",
            metadata={"error": error},
        )
