"""
cadgenesis.transformer.evolution.versioning
============================================
Architecture versioning for the Configurable Transformer Evolution Framework.

An architecture is a *description* of a transformer (a plain dict/JSON).
Versioning associates it with:

* a semantic version (``major.minor.patch``), and
* a content hash — a deterministic fingerprint of the spec, so that *identical
  specs always hash identically* and *different specs almost never collide*.

:class:`VersionedArchitecture` is the mutable, auditable holder: every
``upgrade`` bumps the version, records the previous spec, and recomputes the
hash, producing an immutable history that supports reproducibility and
rollback.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def hash_architecture(spec: dict) -> str:
    """
    SHA-256 content hash of an architecture spec.

    The spec is canonicalised (sorted keys, compact separators) so the hash is
    stable across serialisation orderings.
    """
    if not isinstance(spec, dict):
        raise TypeError("architecture spec must be a dict.")
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ArchitectureVersion:
    """Semantic version of an architecture."""

    major: int = 1
    minor: int = 0
    patch: int = 0

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative int.")

    def bump(self, part: str = "minor") -> ArchitectureVersion:
        """Return a new version with ``major | minor | patch`` incremented."""
        if part not in ("major", "minor", "patch"):
            raise ValueError("part must be one of 'major', 'minor', 'patch'.")
        if part == "major":
            return ArchitectureVersion(self.major + 1, 0, 0)
        if part == "minor":
            return ArchitectureVersion(self.major, self.minor + 1, 0)
        return ArchitectureVersion(self.major, self.minor, self.patch + 1)

    @classmethod
    def parse(cls, text: str) -> ArchitectureVersion:
        if not _VERSION_RE.match(text):
            raise ValueError(f"invalid version string {text!r}; expected 'major.minor.patch'.")
        major, minor, patch = (int(x) for x in text.split("."))
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class VersionedArchitecture:
    """
    A versioned, auditable transformer architecture description.

    Parameters
    ----------
    name : str
        Human-readable architecture name.
    spec : dict
        The architecture description (JSON-serialisable).
    version : ArchitectureVersion | str
        Initial semantic version.
    history : list[dict]
        Prior (version, hash, spec) records; each ``upgrade`` appends one.
    """

    name: str
    spec: dict = field(default_factory=dict)
    version: ArchitectureVersion = field(default_factory=ArchitectureVersion)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("architecture name must be non-empty.")
        if isinstance(self.version, str):
            self.version = ArchitectureVersion.parse(self.version)
        if not isinstance(self.spec, dict):
            raise TypeError("spec must be a dict.")

    # -------------------------------------------------------------- identity

    @property
    def content_hash(self) -> str:
        return hash_architecture(self.spec)

    def full_version(self) -> str:
        """Human-readable identity: ``name@version (hash[:8])``."""
        return f"{self.name}@{self.version} ({self.content_hash[:8]})"

    def fingerprint(self) -> dict:
        """Stable record for experiment registries."""
        return {
            "name": self.name,
            "version": str(self.version),
            "hash": self.content_hash,
        }

    # ---------------------------------------------------------------- mutate

    def upgrade(
        self,
        new_spec: dict,
        bump: str = "minor",
        commit: bool = True,
    ) -> VersionedArchitecture:
        """
        Adopt ``new_spec``, bump the version, and (optionally) record the old
        spec in ``history`` for auditability / rollback.
        """
        if not isinstance(new_spec, dict):
            raise TypeError("new_spec must be a dict.")
        if commit:
            self.history.append(
                {
                    "version": str(self.version),
                    "hash": hash_architecture(self.spec),
                    "spec": self.spec,
                }
            )
        self.spec = new_spec
        self.version = self.version.bump(bump)
        return self

    # ------------------------------------------------------------ serialise

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": str(self.version),
            "content_hash": self.content_hash,
            "spec": self.spec,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> VersionedArchitecture:
        return cls(
            name=payload["name"],
            spec=payload["spec"],
            version=ArchitectureVersion.parse(payload["version"]),
            history=payload.get("history", []),
        )

    def save(self, path: str | Any) -> None:
        """Persist to JSON (``path`` may be a str or ``pathlib.Path``)."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Any) -> VersionedArchitecture:
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
