"""
cadgenesis.collaboration.compute_credits
========================================
Compute Credit System for the CADGenesis-LM Collaborative Research Economy.

Supports GPU credits, CPU credits, storage credits, and compute accounting
for tracking resource usage and allocation.
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

logger = logging.getLogger("cadgenesis.collaboration.compute_credits")


class CreditType:
    """Types of compute credits."""

    GPU = "gpu"
    CPU = "cpu"
    STORAGE = "storage"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass
class CreditBalance:
    """Credit balance for a contributor or project."""

    entity_id: str  # contributor_id or project_id
    gpu_credits: float = 0.0
    cpu_credits: float = 0.0
    storage_credits: float = 0.0  # GB-hours
    memory_credits: float = 0.0  # GB-hours
    network_credits: float = 0.0  # GB transferred
    updated_at: float = field(default_factory=time.time)

    def get(self, credit_type: str) -> float:
        return getattr(self, f"{credit_type}_credits", 0.0)

    def add(self, credit_type: str, amount: float) -> None:
        current = self.get(credit_type)
        setattr(self, f"{credit_type}_credits", current + amount)
        self.updated_at = time.time()

    def deduct(self, credit_type: str, amount: float) -> bool:
        current = self.get(credit_type)
        if current < amount:
            return False
        setattr(self, f"{credit_type}_credits", current - amount)
        self.updated_at = time.time()
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CreditBalance:
        return cls(
            entity_id=str(data["entity_id"]),
            gpu_credits=float(data.get("gpu_credits", 0.0)),
            cpu_credits=float(data.get("cpu_credits", 0.0)),
            storage_credits=float(data.get("storage_credits", 0.0)),
            memory_credits=float(data.get("memory_credits", 0.0)),
            network_credits=float(data.get("network_credits", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass
class ComputeUsageRecord:
    """Record of compute resource usage."""

    id: str
    entity_id: str  # contributor_id or project_id
    credit_type: str
    amount: float  # Credits consumed
    description: str
    job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComputeUsageRecord:
        return cls(
            id=str(data["id"]),
            entity_id=str(data["entity_id"]),
            credit_type=str(data["credit_type"]),
            amount=float(data["amount"]),
            description=str(data["description"]),
            job_id=data.get("job_id"),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", 0.0)),
        )


@dataclass
class ComputeGrant:
    """A grant of compute credits to an entity."""

    id: str
    entity_id: str
    credit_type: str
    amount: float
    granted_by: str
    reason: str
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComputeGrant:
        return cls(
            id=str(data["id"]),
            entity_id=str(data["entity_id"]),
            credit_type=str(data["credit_type"]),
            amount=float(data["amount"]),
            granted_by=str(data["granted_by"]),
            reason=str(data["reason"]),
            expires_at=data.get("expires_at"),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", 0.0)),
        )


# Default credit rates (credits per unit)
DEFAULT_RATES = {
    CreditType.GPU: 1.0,  # 1 credit per GPU-hour
    CreditType.CPU: 0.1,  # 0.1 credits per CPU-hour
    CreditType.STORAGE: 0.01,  # 0.01 credits per GB-hour
    CreditType.MEMORY: 0.05,  # 0.05 credits per GB-hour
    CreditType.NETWORK: 0.001,  # 0.001 credits per GB transferred
}


class ComputeCreditLedger:
    """
    Ledger for compute credit accounting.

    Features:
    - Multi-type credit balances (GPU, CPU, storage, memory, network)
    - Usage tracking with job correlation
    - Grant/expiry management
    - Rate-based billing
    - Audit trail
    """

    BALANCES_FILE = "credit_balances.json"
    USAGE_FILE = "credit_usage.json"
    GRANTS_FILE = "credit_grants.json"

    def __init__(
        self,
        directory: str | os.PathLike[str],
        rates: Mapping[str, float] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rates = dict(DEFAULT_RATES)
        if rates:
            self.rates.update(rates)
        self._lock = threading.RLock()
        self._balances: dict[str, CreditBalance] = {}
        self._usage_records: list[ComputeUsageRecord] = []
        self._grants: list[ComputeGrant] = []
        self._load()

    def _load(self) -> None:
        # Load balances
        balances_path = self.directory / self.BALANCES_FILE
        if balances_path.exists():
            try:
                data = json.loads(balances_path.read_text(encoding="utf-8"))
                for item in data.get("balances", []):
                    balance = CreditBalance.from_dict(item)
                    self._balances[balance.entity_id] = balance
            except (ValueError, OSError) as exc:
                logger.warning("credit balances unreadable: %s", exc)

        # Load usage records
        usage_path = self.directory / self.USAGE_FILE
        if usage_path.exists():
            try:
                data = json.loads(usage_path.read_text(encoding="utf-8"))
                for item in data.get("usage", []):
                    record = ComputeUsageRecord.from_dict(item)
                    self._usage_records.append(record)
            except (ValueError, OSError) as exc:
                logger.warning("credit usage unreadable: %s", exc)

        # Load grants
        grants_path = self.directory / self.GRANTS_FILE
        if grants_path.exists():
            try:
                data = json.loads(grants_path.read_text(encoding="utf-8"))
                for item in data.get("grants", []):
                    grant = ComputeGrant.from_dict(item)
                    self._grants.append(grant)
            except (ValueError, OSError) as exc:
                logger.warning("credit grants unreadable: %s", exc)

    def _persist_balances(self) -> None:
        payload = {"balances": [b.to_dict() for b in self._balances.values()]}
        path = self.directory / self.BALANCES_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".balances-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _persist_usage(self) -> None:
        payload = {"usage": [r.to_dict() for r in self._usage_records]}
        path = self.directory / self.USAGE_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".usage-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _persist_grants(self) -> None:
        payload = {"grants": [g.to_dict() for g in self._grants]}
        path = self.directory / self.GRANTS_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".grants-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _get_or_create_balance(self, entity_id: str) -> CreditBalance:
        with self._lock:
            if entity_id not in self._balances:
                self._balances[entity_id] = CreditBalance(entity_id=entity_id)
            return self._balances[entity_id]

    # ------------------------------------------------------------ balances

    def get_balance(self, entity_id: str) -> CreditBalance:
        """Get credit balance for an entity."""
        with self._lock:
            return self._get_or_create_balance(entity_id)

    def add_credits(
        self,
        entity_id: str,
        credit_type: str,
        amount: float,
        description: str = "",
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CreditBalance:
        """Add credits to an entity's balance."""
        with self._lock:
            balance = self._get_or_create_balance(entity_id)
            balance.add(credit_type, amount)
            self._persist_balances()

            # Record usage (negative amount for tracking)
            record = ComputeUsageRecord(
                id=str(uuid.uuid4()),
                entity_id=entity_id,
                credit_type=credit_type,
                amount=-amount,  # Negative for credit addition
                description=description or f"Credit grant: {amount} {credit_type}",
                job_id=job_id,
                metadata=dict(metadata or {}),
            )
            self._usage_records.append(record)
            self._persist_usage()
            return balance

    def deduct_credits(
        self,
        entity_id: str,
        credit_type: str,
        amount: float,
        description: str = "",
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[bool, CreditBalance]:
        """Deduct credits from an entity's balance. Returns (success, balance)."""
        with self._lock:
            balance = self._get_or_create_balance(entity_id)
            success = balance.deduct(credit_type, amount)
            if success:
                self._persist_balances()
                record = ComputeUsageRecord(
                    id=str(uuid.uuid4()),
                    entity_id=entity_id,
                    credit_type=credit_type,
                    amount=amount,
                    description=description or f"Credit deduction: {amount} {credit_type}",
                    job_id=job_id,
                    metadata=dict(metadata or {}),
                )
                self._usage_records.append(record)
                self._persist_usage()
            return success, balance

    # ------------------------------------------------------------ grants

    def create_grant(
        self,
        entity_id: str,
        credit_type: str,
        amount: float,
        granted_by: str,
        reason: str,
        expires_in_days: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ComputeGrant:
        """Create a compute credit grant."""
        with self._lock:
            expires_at = None
            if expires_in_days is not None:
                expires_at = time.time() + (expires_in_days * 86400)

            grant = ComputeGrant(
                id=str(uuid.uuid4()),
                entity_id=entity_id,
                credit_type=credit_type,
                amount=amount,
                granted_by=granted_by,
                reason=reason,
                expires_at=expires_at,
                metadata=dict(metadata or {}),
            )
            self._grants.append(grant)
            self._persist_grants()

            # Apply the grant immediately
            self.add_credits(entity_id, credit_type, amount, description=f"Grant: {reason}")
            logger.info(
                "created grant %s for %s: %.2f %s", grant.id, entity_id, amount, credit_type
            )
            return grant

    def get_active_grants(self, entity_id: str) -> list[ComputeGrant]:
        """Get active (non-expired) grants for an entity."""
        with self._lock:
            now = time.time()
            return [
                g
                for g in self._grants
                if g.entity_id == entity_id and (g.expires_at is None or g.expires_at > now)
            ]

    def process_expired_grants(self) -> int:
        """Process expired grants (claw back credits if configured)."""
        with self._lock:
            now = time.time()
            expired = [g for g in self._grants if g.expires_at is not None and g.expires_at <= now]
            for grant in expired:
                # Optionally claw back - for now just mark as expired
                logger.info("grant %s expired", grant.id)
            return len(expired)

    # ------------------------------------------------------------ usage & billing

    def record_usage(
        self,
        entity_id: str,
        credit_type: str,
        units: float,  # e.g., GPU-hours, CPU-hours, GB-hours
        description: str,
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[bool, CreditBalance]:
        """
        Record resource usage and deduct appropriate credits.

        Uses configured rates to convert units to credits.
        """
        rate = self.rates.get(credit_type, 1.0)
        credits = units * rate
        return self.deduct_credits(entity_id, credit_type, credits, description, job_id, metadata)

    def get_usage_history(
        self,
        entity_id: str,
        credit_type: str | None = None,
        since: float | None = None,
        limit: int = 1000,
    ) -> list[ComputeUsageRecord]:
        """Get usage history for an entity."""
        with self._lock:
            records = [r for r in self._usage_records if r.entity_id == entity_id]
            if credit_type:
                records = [r for r in records if r.credit_type == credit_type]
            if since:
                records = [r for r in records if r.created_at >= since]
            records.sort(key=lambda r: r.created_at, reverse=True)
            return records[:limit]

    def get_usage_summary(
        self,
        entity_id: str,
        since: float | None = None,
    ) -> dict[str, Any]:
        """Get aggregated usage summary by credit type."""
        with self._lock:
            records = [r for r in self._usage_records if r.entity_id == entity_id and r.amount > 0]
            if since:
                records = [r for r in records if r.created_at >= since]

            summary = {}
            for record in records:
                ct = record.credit_type
                if ct not in summary:
                    summary[ct] = {"total_credits": 0.0, "total_units": 0.0, "record_count": 0}
                # Convert credits back to units using rate
                rate = self.rates.get(ct, 1.0)
                units = record.amount / rate if rate > 0 else 0
                summary[ct]["total_credits"] += record.amount
                summary[ct]["total_units"] += units
                summary[ct]["record_count"] += 1

            return summary

    def estimate_cost(
        self,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
        storage_gb_hours: float = 0.0,
        memory_gb_hours: float = 0.0,
        network_gb: float = 0.0,
    ) -> dict[str, float]:
        """Estimate compute cost for a workload."""
        return {
            CreditType.GPU: gpu_hours * self.rates.get(CreditType.GPU, 1.0),
            CreditType.CPU: cpu_hours * self.rates.get(CreditType.CPU, 0.1),
            CreditType.STORAGE: storage_gb_hours * self.rates.get(CreditType.STORAGE, 0.01),
            CreditType.MEMORY: memory_gb_hours * self.rates.get(CreditType.MEMORY, 0.05),
            CreditType.NETWORK: network_gb * self.rates.get(CreditType.NETWORK, 0.001),
        }

    # ------------------------------------------------------------ admin

    def list_all_balances(self) -> list[CreditBalance]:
        """List all credit balances."""
        with self._lock:
            return list(self._balances.values())

    def get_total_credits_issued(self) -> dict[str, float]:
        """Get total credits issued by type."""
        with self._lock:
            totals = {ct: 0.0 for ct in self.rates}
            for record in self._usage_records:
                if record.amount < 0:  # Credit additions
                    totals[record.credit_type] = totals.get(record.credit_type, 0.0) + abs(
                        record.amount
                    )
            return totals

    def get_total_credits_consumed(self) -> dict[str, float]:
        """Get total credits consumed by type."""
        with self._lock:
            totals = {ct: 0.0 for ct in self.rates}
            for record in self._usage_records:
                if record.amount > 0:  # Deductions
                    totals[record.credit_type] = totals.get(record.credit_type, 0.0) + record.amount
            return totals


__all__ = [
    "DEFAULT_RATES",
    "ComputeCreditLedger",
    "ComputeGrant",
    "ComputeUsageRecord",
    "CreditBalance",
    "CreditType",
]
