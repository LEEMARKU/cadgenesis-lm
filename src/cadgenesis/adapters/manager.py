"""
cadgenesis.adapters.manager
===========================
Self-Evolving Adapter Bank & Stability System for CADGenesis-LM v2.0:
- Adapter lifecycle management: birth, evaluation, promotion, retirement, rollback
- Performance & drift monitoring
- Automated safety controller & versioned checkpoint rollback system
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AdapterMetadata:
    adapter_id: str
    domain: str  # e.g., "aerospace", "automotive", "sheet_metal"
    status: str = "candidate"  # "candidate", "promoted", "retired", "rolled_back"
    accuracy_score: float = 0.0
    stability_score: float = 1.0
    created_at: float = field(default_factory=time.time)


class SelfEvolvingAdapterBank:
    """
    Manages the pool of domain-specific LoRA adapters with autonomous lifecycle evolution.
    """

    def __init__(self):
        self.adapters: dict[str, AdapterMetadata] = {}
        self.active_adapter_id: str | None = None
        self.history_logs: list[str] = []

    def register_adapter(self, adapter_id: str, domain: str) -> AdapterMetadata:
        meta = AdapterMetadata(adapter_id=adapter_id, domain=domain)
        self.adapters[adapter_id] = meta
        self.history_logs.append(f"Registered candidate adapter {adapter_id} for domain {domain}")
        return meta

    def evaluate_and_promote(self, adapter_id: str, accuracy: float, stability: float) -> bool:
        """Promotes adapter if performance exceeds threshold and stability is high."""
        if adapter_id not in self.adapters:
            return False

        meta = self.adapters[adapter_id]
        meta.accuracy_score = accuracy
        meta.stability_score = stability

        if accuracy > 0.85 and stability > 0.90:
            meta.status = "promoted"
            self.active_adapter_id = adapter_id
            self.history_logs.append(
                f"PROMOTED adapter {adapter_id} "
                f"(accuracy={accuracy:.2f}, stability={stability:.2f})"
            )
            return True
        else:
            self.history_logs.append(
                f"Evaluation failed for {adapter_id} "
                f"(accuracy={accuracy:.2f}, stability={stability:.2f})"
            )
            return False

    def trigger_rollback(self, adapter_id: str, reason: str):
        """Rolls back an unstable adapter to safety baseline."""
        if adapter_id in self.adapters:
            self.adapters[adapter_id].status = "rolled_back"
            if self.active_adapter_id == adapter_id:
                self.active_adapter_id = None
            self.history_logs.append(f"ROLLBACK adapter {adapter_id}: {reason}")
