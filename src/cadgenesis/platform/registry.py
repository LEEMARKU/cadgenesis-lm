"""
cadgenesis.platform.registry
============================
Model registry for the CADGenesis-LM production platform.

- Versioning: semver-style model versions per name (``v1``, ``v2.1.0``...)
- Metadata: params/metrics/arch/config hash/tag lineage
- Rollback: promote any previously registered version back to production
- Deployment history: append-only log of deployment events per version
- Storage: JSON index + optional checksum verification (``utils.hashing``)

Pure-Python, filesystem-backed; safe for multi-process readers via atomic
rewrites.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.platform.registry")

INDEX_FILE = "index.json"
METADATA_FILE = "metadata.json"
CHECKPOINT_PATTERNS = ("*.pt", "*.bin", "*.safetensors")


def parse_version(version: str) -> tuple[int, ...]:
    parts = []
    for segment in version.lstrip("v").replace("-", ".").split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break
    return tuple(parts) or (0,)


@dataclass
class ModelVersion:
    """One registered model version."""

    name: str
    version: str
    path: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelVersion:
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            path=str(data["path"]),
            created_at=float(data.get("created_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
            checksum=data.get("checksum"),
        )


@dataclass
class DeploymentRecord:
    """One deployment event in the history."""

    name: str
    version: str
    environment: str
    timestamp: float = field(default_factory=time.time)
    actor: str = "system"
    status: str = "deployed"  # deployed | rolled_back | promoted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Versioned model registry with rollback and deployment history."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index_path = self.directory / INDEX_FILE
        self._versions: dict[str, dict[str, ModelVersion]] = {}  # name -> version -> record
        self._aliases: dict[str, dict[str, str]] = {}  # name -> alias -> version
        self._deployments: dict[str, list[DeploymentRecord]] = {}
        self._lock = threading.RLock()
        self._load()

    # ---------------------------------------------------------------- index

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.warning("registry index unreadable, starting empty: %s", exc)
            return
        self._versions = {
            name: {ver: ModelVersion.from_dict(v) for ver, v in versions.items()}
            for name, versions in data.get("versions", {}).items()
        }
        self._aliases = {
            name: {alias: version for alias, version in aliases.items()}
            for name, aliases in data.get("aliases", {}).items()
        }
        self._deployments = {
            name: [DeploymentRecord(**d) for d in history]
            for name, history in data.get("deployments", {}).items()
        }

    def _persist(self) -> None:
        payload = {
            "versions": {
                name: {ver: v.to_dict() for ver, v in versions.items()}
                for name, versions in self._versions.items()
            },
            "aliases": self._aliases,
            "deployments": {
                name: [d.to_dict() for d in history] for name, history in self._deployments.items()
            },
        }
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".index-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, self._index_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------------ versioning

    def register(
        self,
        name: str,
        path: str | os.PathLike[str],
        version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        checksum: str | None = None,
    ) -> ModelVersion:
        """Register a model checkpoint. Default version is the next vN."""
        with self._lock:
            version = version or self._next_version(name)
            versions = self._versions.setdefault(name, {})
            if version in versions:
                raise ValueError(f"version {name}@{version} already registered")
            record = ModelVersion(
                name=name,
                version=version,
                path=str(path),
                metadata=dict(metadata or {}),
                checksum=checksum,
            )
            versions[version] = record
            self._aliases.setdefault(name, {})["latest"] = version
            self._persist()
            logger.info("registered %s@%s -> %s", name, version, record.path)
            return record

    def _next_version(self, name: str) -> str:
        versions = list(self._versions.get(name, {}).keys())
        if not versions:
            return "v1"
        highest = max(parse_version(v) for v in versions)
        return f"v{highest[0] + 1}"

    def get(
        self, name: str, version: str | None = None, alias: str | None = None
    ) -> ModelVersion | None:
        """Fetch a version; resolve ``alias`` (latest/production) or default latest."""
        with self._lock:
            versions = self._versions.get(name, {})
            if not versions:
                return None
            if version is not None:
                return versions.get(version)
            alias = alias or "latest"
            aliases = self._aliases.get(name, {})
            resolved = aliases.get(alias)
            if resolved is None:
                resolved = max(versions, key=lambda v: parse_version(v))
            return versions.get(resolved)

    def list_versions(self, name: str) -> list[ModelVersion]:
        with self._lock:
            return [
                v
                for _, v in sorted(
                    self._versions.get(name, {}).items(), key=lambda kv: parse_version(kv[0])
                )
            ]

    def list_models(self) -> list[str]:
        with self._lock:
            return sorted(self._versions)

    # ---------------------------------------------------------- aliases/ops

    def set_alias(self, name: str, alias: str, version: str) -> None:
        with self._lock:
            if name not in self._versions or version not in self._versions[name]:
                raise ValueError(f"unknown version {name}@{version}")
            self._aliases.setdefault(name, {})[alias] = version
            self._persist()

    def promote(
        self, name: str, version: str, environment: str = "production", actor: str = "system"
    ) -> ModelVersion:
        """Rollback/promote: point an environment alias at a registered version."""
        record = self.get(name, version=version)
        if record is None:
            raise ValueError(f"unknown version {name}@{version}")
        with self._lock:
            self._aliases.setdefault(name, {})[environment] = version
            self._deployments.setdefault(name, []).append(
                DeploymentRecord(
                    name=name,
                    version=version,
                    environment=environment,
                    actor=actor,
                    status="promoted",
                )
            )
            self._persist()
            logger.info("promoted %s@%s to %s", name, version, environment)
            return record

    def rollback(
        self, name: str, environment: str = "production", actor: str = "system"
    ) -> ModelVersion | None:
        """Roll back an environment to its previous deployment."""
        with self._lock:
            history = [
                d
                for d in self._deployments.get(name, [])
                if d.environment == environment and d.status == "promoted"
            ]
            if len(history) < 2:
                return None
            previous = history[-2]
            self._aliases.setdefault(name, {})[environment] = previous.version
            self._deployments.setdefault(name, []).append(
                DeploymentRecord(
                    name=name,
                    version=previous.version,
                    environment=environment,
                    actor=actor,
                    status="rolled_back",
                )
            )
            self._persist()
            logger.info("rolled back %s to %s@%s", environment, name, previous.version)
            return self.get(name, version=previous.version)

    def deployment_history(self, name: str, limit: int = 50) -> list[DeploymentRecord]:
        with self._lock:
            return list(self._deployments.get(name, []))[-limit:]

    def export_index(self) -> dict[str, Any]:
        with self._lock:
            return {
                "versions": {
                    n: {v: m.to_dict() for v, m in vs.items()} for n, vs in self._versions.items()
                },
                "aliases": {n: dict(a) for n, a in self._aliases.items()},
            }


__all__ = ["DeploymentRecord", "ModelRegistry", "ModelVersion", "parse_version"]
