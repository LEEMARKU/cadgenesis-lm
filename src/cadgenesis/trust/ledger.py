"""Ledger for experiments and federated training."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .core import RecordType, TrustConfig, TrustLayer
from .provenance import ExperimentProvenance


@dataclass
class ExperimentEntry:
    """An entry in the experiment ledger."""

    experiment_id: str
    provenance: ExperimentProvenance
    trust_record_id: str | None = None
    created_at: float = field(default_factory=time.time)


class ExperimentLedger:
    """Ledger for tracking all experiments with cryptographic integrity."""

    def __init__(self, trust_layer: TrustLayer | None = None, config: TrustConfig | None = None):
        self.trust_layer = trust_layer or TrustLayer(config or TrustConfig())
        self._entries: dict[str, ExperimentEntry] = {}
        self._lock = RLock()

    def create_experiment(
        self,
        name: str,
        hypothesis: str,
        configuration: dict[str, Any],
        hyperparameters: dict[str, Any],
        dataset_hashes: list[str],
        model_hashes: list[str],
        created_by: str,
    ) -> ExperimentProvenance:
        """Create a new experiment provenance record."""
        provenance = ExperimentProvenance(
            provenance_id=str(uuid.uuid4()),
            name=name,
            version="1.0.0",
            content_hash="",  # Will be computed on first content
            created_at=time.time(),
            created_by=created_by,
            hypothesis=hypothesis,
            configuration=configuration,
            hyperparameters=hyperparameters,
            dataset_hashes=dataset_hashes,
            model_hashes=model_hashes,
        )

        # Create trust record
        trust_record = self.trust_layer.create_record(
            RecordType.EXPERIMENT,
            provenance.__dict__,
            metadata={"experiment_id": provenance.provenance_id},
        )

        entry = ExperimentEntry(
            experiment_id=provenance.provenance_id,
            provenance=provenance,
            trust_record_id=trust_record.record_id,
        )
        with self._lock:
            self._entries[provenance.provenance_id] = entry

        return provenance

    def get_experiment(self, experiment_id: str) -> ExperimentProvenance | None:
        with self._lock:
            entry = self._entries.get(experiment_id)
            return entry.provenance if entry else None

    def update_experiment(self, experiment_id: str, provenance: ExperimentProvenance) -> bool:
        with self._lock:
            entry = self._entries.get(experiment_id)
            if not entry:
                return False
            entry.provenance = provenance
            # Create new trust record for update
            trust_record = self.trust_layer.create_record(
                RecordType.EXPERIMENT,
                provenance.__dict__,
                metadata={"experiment_id": experiment_id, "update": True},
            )
            entry.trust_record_id = trust_record.record_id
            return True

    def list_experiments(
        self, status: str | None = None, limit: int = 100
    ) -> list[ExperimentProvenance]:
        with self._lock:
            experiments = [e.provenance for e in self._entries.values()]
            if status:
                experiments = [e for e in experiments if e.status == status]
            return experiments[:limit]

    def verify_experiment(self, experiment_id: str) -> tuple[bool, list[str]]:
        """Verify the integrity of an experiment's provenance chain."""
        with self._lock:
            entry = self._entries.get(experiment_id)
            if not entry:
                return False, ["Experiment not found"]

            trust_record_id = entry.trust_record_id
            if trust_record_id is None:
                return False, ["Trust record not found"]

            trust_record = self.trust_layer.get_record(trust_record_id)
            if not trust_record:
                return False, ["Trust record not found"]

            valid = self.trust_layer.verify_record(trust_record)
            return valid, [] if valid else ["Signature verification failed"]


@dataclass
class FederatedRound:
    """A single round in federated training."""

    round_id: str
    round_number: int
    participants: list[str]
    aggregation_method: str
    global_model_hash: str
    participant_updates: dict[str, dict[str, Any]]  # node_id -> {model_hash, metrics, etc.}
    timestamp: float
    trust_record_id: str | None = None


@dataclass
class FederatedTrainingLedger:
    """Ledger for federated training rounds."""

    experiment_id: str
    trust_layer: TrustLayer
    rounds: list[FederatedRound] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock)

    def add_round(
        self,
        round_number: int,
        participants: list[str],
        aggregation_method: str,
        global_model_hash: str,
        participant_updates: dict[str, dict[str, Any]],
    ) -> FederatedRound:
        with self._lock:
            round_id = str(uuid.uuid4())
            round_obj = FederatedRound(
                round_id=round_id,
                round_number=round_number,
                participants=participants,
                aggregation_method=aggregation_method,
                global_model_hash=global_model_hash,
                participant_updates=participant_updates,
                timestamp=time.time(),
            )

            # Create trust record
            trust_record = self.trust_layer.create_record(
                RecordType.FEDERATED_ROUND,
                {
                    "round_id": round_id,
                    "round_number": round_number,
                    "participants": participants,
                    "aggregation_method": aggregation_method,
                    "global_model_hash": global_model_hash,
                    "participant_updates": participant_updates,
                    "timestamp": round_obj.timestamp,
                },
                metadata={"experiment_id": self.experiment_id},
            )
            round_obj.trust_record_id = trust_record.record_id
            self.rounds.append(round_obj)
            return round_obj

    def get_round(self, round_number: int) -> FederatedRound | None:
        with self._lock:
            for r in self.rounds:
                if r.round_number == round_number:
                    return r
            return None

    def get_latest_round(self) -> FederatedRound | None:
        with self._lock:
            return self.rounds[-1] if self.rounds else None

    def verify_round(self, round_number: int) -> tuple[bool, list[str]]:
        with self._lock:
            round_obj = self.get_round(round_number)
            if not round_obj or not round_obj.trust_record_id:
                return False, ["Round not found"]

            trust_record = self.trust_layer.get_record(round_obj.trust_record_id)
            if not trust_record:
                return False, ["Trust record not found"]

            valid = self.trust_layer.verify_record(trust_record)
            return valid, [] if valid else ["Signature verification failed"]

    def verify_all_rounds(self) -> tuple[bool, list[str]]:
        with self._lock:
            errors = []
            for r in self.rounds:
                valid, errs = self.verify_round(r.round_number)
                if not valid:
                    errors.extend([f"Round {r.round_number}: {e}" for e in errs])
            return len(errors) == 0, errors

    def export_ledger(self) -> dict[str, Any]:
        with self._lock:
            return {
                "experiment_id": self.experiment_id,
                "rounds": [
                    {
                        "round_id": r.round_id,
                        "round_number": r.round_number,
                        "participants": r.participants,
                        "aggregation_method": r.aggregation_method,
                        "global_model_hash": r.global_model_hash,
                        "participant_updates": r.participant_updates,
                        "timestamp": r.timestamp,
                        "trust_record_id": r.trust_record_id,
                    }
                    for r in self.rounds
                ],
            }
