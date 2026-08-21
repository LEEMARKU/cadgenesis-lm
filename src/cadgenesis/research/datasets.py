"""
cadgenesis.research.datasets
============================
Dataset versioning for CADGenesis-LM research infrastructure.

- Semantic versioning: ``MAJOR.MINOR.PATCH`` per dataset
- Lineage: parent dataset links recorded on every version
- Snapshots: immutable copies (files + SHA-256 manifest)
- Rollback: re-point ``latest``/an environment alias to an older version

Layout::

    <root>/<dataset_name>/versions/v1.0.0/   (manifest.json + files)
    <root>/<dataset_name>/index.json         (aliases + lineage)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.research.datasets")

INDEX_FILE = "index.json"
VERSIONS_DIR = "versions"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bump_version(version: str, part: str = "patch") -> str:
    major, minor, patch = (int(p) for p in version.split(".")[:3])
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


@dataclass
class DatasetVersion:
    """One immutable dataset snapshot."""

    name: str
    version: str
    path: str
    sha256: str
    created_at: float = field(default_factory=time.time)
    parent: str | None = None  # parent version string (lineage)
    records: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetVersion:
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            created_at=float(data.get("created_at", 0.0)),
            parent=data.get("parent"),
            records=int(data.get("records", 0)),
            metadata=dict(data.get("metadata", {})),
        )


class DatasetRegistry:
    """Semantic versioning, lineage, snapshots and rollback for datasets."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._versions: dict[str, dict[str, DatasetVersion]] = {}
        self._aliases: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        index = self.root / INDEX_FILE
        if not index.exists():
            return
        data = json.loads(index.read_text(encoding="utf-8"))
        self._versions = {
            name: {ver: DatasetVersion.from_dict(v) for ver, v in versions.items()}
            for name, versions in data.get("versions", {}).items()
        }
        self._aliases = {name: dict(a) for name, a in data.get("aliases", {}).items()}

    def _persist(self) -> None:
        payload = {
            "versions": {
                n: {v: d.to_dict() for v, d in vs.items()} for n, vs in self._versions.items()
            },
            "aliases": self._aliases,
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".index-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, self.root / INDEX_FILE)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------------ snapshot

    def snapshot(
        self,
        name: str,
        source: str | os.PathLike[str],
        version: str | None = None,
        parent: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        """Snapshot files from ``source`` into a new immutable version.

        ``source`` may be a file or directory; a SHA-256 manifest is stored
        alongside the copy.  ``parent`` records lineage (defaults to the
        latest version of the same dataset).
        """
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"dataset source not found: {source}")
        with self._lock:
            versions = self._versions.setdefault(name, {})
            target_version = version or bump_version(
                max(versions, key=lambda v: [int(p) for p in v.split(".")[:3]])
                if versions
                else "0.0.0"
            )
            if target_version in versions:
                raise ValueError(f"version {name}@{target_version} already exists")
            if parent is None:
                parent = self._aliases.get(name, {}).get("latest")
            version_dir = self.root / name / VERSIONS_DIR / target_version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            version_dir.mkdir(parents=True, exist_ok=True)
            manifest: dict[str, str] = {}
            if source_path.is_dir():
                for file in sorted(source_path.rglob("*")):
                    if file.is_file():
                        rel = file.relative_to(source_path).as_posix()
                        target = version_dir / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, target)
                        manifest[rel] = sha256_file(target)
            else:
                target = version_dir / source_path.name
                shutil.copy2(source_path, target)
                manifest[source_path.name] = sha256_file(target)
            manifest_path = version_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            record = DatasetVersion(
                name=name,
                version=target_version,
                path=str(version_dir),
                sha256=sha256_file(manifest_path),
                parent=parent,
                records=len(manifest),
                metadata=dict(metadata or {}),
            )
            versions[target_version] = record
            self._aliases.setdefault(name, {})["latest"] = target_version
            self._persist()
            logger.info(
                "dataset %s@%s snapshotted (%d files, parent=%s)",
                name,
                target_version,
                len(manifest),
                parent,
            )
            return record

    # -------------------------------------------------------------- access

    def get(
        self, name: str, version: str | None = None, alias: str = "latest"
    ) -> DatasetVersion | None:
        with self._lock:
            versions = self._versions.get(name, {})
            if not versions:
                return None
            if version is not None:
                return versions.get(version)
            resolved = self._aliases.get(name, {}).get(alias)
            if resolved is None:
                resolved = max(versions, key=lambda v: [int(p) for p in v.split(".")[:3]])
            return versions.get(resolved)

    def lineage(self, name: str, version: str | None = None) -> list[DatasetVersion]:
        """Walk the parent chain from ``version`` (newest first)."""
        chain: list[DatasetVersion] = []
        current = self.get(name, version=version)
        seen: set[str] = set()
        while current is not None and current.version not in seen:
            seen.add(current.version)
            chain.append(current)
            parent = current.parent
            current = self._versions.get(name, {}).get(parent) if parent else None
        return chain

    def list_versions(self, name: str) -> list[DatasetVersion]:
        with self._lock:
            versions = self._versions.get(name, {})
            return [
                versions[v]
                for v in sorted(versions, key=lambda x: [int(p) for p in x.split(".")[:3]])
            ]

    def list_datasets(self) -> list[str]:
        with self._lock:
            return sorted(self._versions)

    # ------------------------------------------------------- rollback/alias

    def rollback(self, name: str, version: str, alias: str = "latest") -> DatasetVersion:
        """Re-point an alias (e.g. ``latest``) at an older snapshot."""
        with self._lock:
            record = self._require(name, version)
            self._aliases.setdefault(name, {})[alias] = record.version
            self._persist()
            logger.info("dataset %s alias %s -> %s", name, alias, record.version)
            return record

    def _require(self, name: str, version: str) -> DatasetVersion:
        record = self._versions.get(name, {}).get(version)
        if record is None:
            raise KeyError(f"unknown dataset version {name}@{version}")
        return record

    def verify(self, name: str, version: str) -> bool:
        """Recompute the manifest hashes to confirm snapshot integrity."""
        record = self._require(name, version)
        version_dir = Path(record.path)
        manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
        return all(sha256_file(version_dir / rel) == expected for rel, expected in manifest.items())


__all__ = ["DatasetRegistry", "DatasetVersion", "bump_version", "sha256_file"]
