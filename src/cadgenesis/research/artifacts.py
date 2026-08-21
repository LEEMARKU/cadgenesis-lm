"""
cadgenesis.research.artifacts
=============================
Artifact registry for CADGenesis-LM research infrastructure.

Stores experiment artifacts (checkpoints, plots, logs, reports) with
content-addressed hashing, metadata and retrieval by name/experiment.

Layout: ``<root>/<experiment_id>/<artifact_name>`` + ``index.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.research.artifacts")


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Content hash of a file (streamed, chunked)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ArtifactRecord:
    """Metadata for one stored artifact."""

    name: str
    experiment_id: str
    path: str  # absolute path of stored copy
    sha256: str
    size_bytes: int
    created_at: float
    kind: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ArtifactRegistry:
    """Store and retrieve artifacts per experiment."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ArtifactRecord] = {}
        self._load_index()

    # -------------------------------------------------------------- storage

    def store(
        self,
        experiment_id: str,
        source: str | os.PathLike[str],
        name: str | None = None,
        kind: str = "file",
        metadata: Mapping[str, Any] | None = None,
        copy: bool = False,
    ) -> ArtifactRecord:
        """Register a file as an artifact. With ``copy=False`` it is moved."""
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        name = name or source.name
        dest_dir = self.root / experiment_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if copy:
            shutil.copy2(source, dest)
        else:
            shutil.move(str(source), dest)
        record = ArtifactRecord(
            name=name,
            experiment_id=experiment_id,
            path=str(dest.resolve()),
            sha256=file_sha256(dest),
            size_bytes=dest.stat().st_size,
            created_at=time.time(),
            kind=kind,
            metadata=dict(metadata or {}),
        )
        key = self._key(experiment_id, name)
        self._records[key] = record
        self._save_index()
        logger.info("stored artifact %s for experiment %s", name, experiment_id)
        return record

    def store_bytes(
        self,
        experiment_id: str,
        name: str,
        data: bytes,
        kind: str = "file",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Store raw bytes directly (e.g. generated plots/reports)."""
        dest_dir = self.root / experiment_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        dest.write_bytes(data)
        record = ArtifactRecord(
            name=name,
            experiment_id=experiment_id,
            path=str(dest.resolve()),
            sha256=file_sha256(dest),
            size_bytes=len(data),
            created_at=time.time(),
            kind=kind,
            metadata=dict(metadata or {}),
        )
        self._records[self._key(experiment_id, name)] = record
        self._save_index()
        return record

    # ----------------------------------------------------------- retrieval

    @staticmethod
    def _key(experiment_id: str, name: str) -> str:
        return f"{experiment_id}/{name}"

    def get(self, experiment_id: str, name: str) -> ArtifactRecord | None:
        return self._records.get(self._key(experiment_id, name))

    def list(self, experiment_id: str | None = None) -> list[ArtifactRecord]:
        records = list(self._records.values())
        if experiment_id is not None:
            records = [r for r in records if r.experiment_id == experiment_id]
        return sorted(records, key=lambda r: r.created_at)

    def read_bytes(self, experiment_id: str, name: str) -> bytes | None:
        record = self.get(experiment_id, name)
        if record is None:
            return None
        return Path(record.path).read_bytes()

    def path(self, experiment_id: str, name: str) -> str | None:
        record = self.get(experiment_id, name)
        return record.path if record else None

    def verify(self, experiment_id: str, name: str) -> bool:
        """Re-hash the stored file and confirm integrity."""
        record = self.get(experiment_id, name)
        if record is None or not Path(record.path).is_file():
            return False
        return file_sha256(record.path) == record.sha256

    def delete(self, experiment_id: str, name: str) -> bool:
        key = self._key(experiment_id, name)
        record = self._records.pop(key, None)
        if record is None:
            return False
        with suppress(OSError):
            os.remove(record.path)
        self._save_index()
        return True

    # --------------------------------------------------------------- index

    def _load_index(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            for entry in payload.get("artifacts", []):
                record = ArtifactRecord.from_dict(entry)
                self._records[self._key(record.experiment_id, record.name)] = record
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("could not load artifact index %s: %s", self.index_path, exc)

    def _save_index(self) -> None:
        payload = {
            "generated": time.time(),
            "artifacts": [r.to_dict() for r in self._records.values()],
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)


__all__ = ["ArtifactRecord", "ArtifactRegistry", "file_sha256"]
