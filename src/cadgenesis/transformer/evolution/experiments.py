"""
cadgenesis.transformer.evolution.experiments
============================================
Experiment registry for the Configurable Transformer Evolution Framework.

Reproducible research requires that an experiment record ties together the
*architecture version*, the *configuration hash* and the *measured metric*.
:class:`ExperimentRegistry` stores these records in memory and can persist them
to / load them from a JSON file.  A config hash is computed with the same
canonical JSON hashing used by architecture versioning, so a record is fully
traceable to the exact config and architecture that produced it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cadgenesis.transformer.evolution.versioning import hash_architecture

logger = logging.getLogger(__name__)


def _parse_record(item: dict) -> ExperimentRecord | None:
    """Parse one persisted record; return None (and log) when malformed."""
    try:
        return ExperimentRecord.from_dict(item)
    except TypeError:
        logger.warning("skipping malformed record: %s", item)
        return None


@dataclass
class ExperimentRecord:
    """One reproducible experiment result."""

    name: str
    arch_version: str  # e.g. "hierarchical-v2@1.1.0 (ab12cd34)"
    config_hash: str
    metric: float
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> ExperimentRecord:
        return cls(**payload)


class ExperimentRegistry:
    """
    In-memory + JSON-persisted experiment store.

    Parameters
    ----------
    path : str | Path | None
        Optional JSON file to seed from / persist to.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: dict[str, ExperimentRecord] = {}
        if self.path is not None and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ core

    def log(
        self,
        name: str,
        *,
        arch_version: str,
        config_hash: str,
        metric: float,
        metadata: dict | None = None,
    ) -> ExperimentRecord:
        """Record a new experiment (replaces any existing record with same name)."""
        if not name.strip():
            raise ValueError("experiment name must be non-empty.")
        record = ExperimentRecord(
            name=name,
            arch_version=arch_version,
            config_hash=config_hash,
            metric=float(metric),
            metadata=metadata or {},
        )
        self._records[name] = record
        logger.info("logged experiment %s (metric=%.4f)", name, metric)
        return record

    def get(self, name: str) -> ExperimentRecord:
        if name not in self._records:
            raise KeyError(f"no experiment named {name!r}.")
        return self._records[name]

    def all(self) -> list[ExperimentRecord]:
        return list(self._records.values())

    def best(self, higher_is_better: bool = True) -> ExperimentRecord | None:
        """Return the best-metric record (None when empty)."""
        if not self._records:
            return None
        return (
            max(self._records.values(), key=lambda r: r.metric)
            if higher_is_better
            else min(self._records.values(), key=lambda r: r.metric)
        )

    def by_arch_version(self, arch_version: str) -> list[ExperimentRecord]:
        return [r for r in self._records.values() if r.arch_version == arch_version]

    # ------------------------------------------------------------- persist

    def save(self, path: str | Path | None = None) -> None:
        """Persist all records to JSON (defaults to ``self.path``)."""
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no persistence path configured.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([r.to_dict() for r in self.all()], indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self.path is not None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
            logger.warning("failed to load experiment registry %s: %s", self.path, exc)
            return
        if isinstance(payload, list):
            for item in payload:
                record = _parse_record(item)
                if record is not None:
                    self._records[record.name] = record

    # ------------------------------------------------------------- summary

    def summary(self) -> dict:
        """Aggregate statistics over the stored records."""
        records = self.all()
        best = self.best() if records else None
        worst = self.best(higher_is_better=False) if records else None
        return {
            "count": len(records),
            "arch_versions": sorted({r.arch_version for r in records}),
            "best": best.to_dict() if best is not None else None,
            "worst": worst.to_dict() if worst is not None else None,
            "config_hashes": len({r.config_hash for r in records}),
        }

    @classmethod
    def from_config_dict(
        cls,
        config: dict,
        registry: ExperimentRegistry | None = None,
    ) -> str:
        """
        Compute the config hash for a configuration dict (convenience helper
        for recording reproducible experiments).
        """
        return hash_architecture(config)
