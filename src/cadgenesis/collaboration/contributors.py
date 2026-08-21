"""
cadgenesis.collaboration.contributors
=====================================
Contributor Registry for the CADGenesis-LM Collaborative Research Economy.

Tracks developers, researchers, reviewers, and organizations with their
affiliations, roles, and contribution history.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cadgenesis.collaboration.contributors")


@dataclass(frozen=True)
class ContributorRole:
    """Contributor role types."""

    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    MAINTAINER = "maintainer"
    ORGANIZATION = "organization"


@dataclass
class Contributor:
    """A contributor to the CADGenesis ecosystem."""

    id: str
    name: str
    email: str
    role: str = ContributorRole.DEVELOPER
    organization: str | None = None
    affiliations: tuple[str, ...] = ()
    github_username: str | None = None
    orcid: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["affiliations"] = list(self.affiliations)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Contributor:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            email=str(data["email"]),
            role=str(data.get("role", ContributorRole.DEVELOPER)),
            organization=data.get("organization"),
            affiliations=tuple(data.get("affiliations", [])),
            github_username=data.get("github_username"),
            orcid=data.get("orcid"),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ContributionRecord:
    """A record of a contribution made by a contributor."""

    id: str
    contributor_id: str
    contribution_type: str  # code, dataset, adapter, plugin, benchmark, review, documentation
    target_id: str  # ID of the thing contributed to (dataset name, plugin name, etc.)
    description: str
    impact_score: float = 0.0  # Computed by reputation system
    verified: bool = False
    verifier_id: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContributionRecord:
        return cls(
            id=str(data["id"]),
            contributor_id=str(data["contributor_id"]),
            contribution_type=str(data["contribution_type"]),
            target_id=str(data["target_id"]),
            description=str(data["description"]),
            impact_score=float(data.get("impact_score", 0.0)),
            verified=bool(data.get("verified", False)),
            verifier_id=data.get("verifier_id"),
            created_at=float(data.get("created_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


class ContributorRegistry:
    """
    Registry for contributors and their contributions.

    Features:
    - CRUD operations for contributors
    - Contribution tracking with verification
    - Role and affiliation management
    - Search and filtering
    - Export/import for portability
    """

    INDEX_FILE = "contributors.json"
    CONTRIBUTIONS_FILE = "contributions.json"

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._contributors: dict[str, Contributor] = {}
        self._contributions: dict[str, list[ContributionRecord]] = {}  # contributor_id -> list
        self._load()

    def _load(self) -> None:
        # Load contributors
        index_path = self.directory / self.INDEX_FILE
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                for item in data.get("contributors", []):
                    contributor = Contributor.from_dict(item)
                    self._contributors[contributor.id] = contributor
            except (ValueError, OSError) as exc:
                logger.warning("contributor index unreadable, starting empty: %s", exc)

        # Load contributions
        contrib_path = self.directory / self.CONTRIBUTIONS_FILE
        if contrib_path.exists():
            try:
                data = json.loads(contrib_path.read_text(encoding="utf-8"))
                for item in data.get("contributions", []):
                    record = ContributionRecord.from_dict(item)
                    self._contributions.setdefault(record.contributor_id, []).append(record)
            except (ValueError, OSError) as exc:
                logger.warning("contributions index unreadable, starting empty: %s", exc)

    def _persist_contributors(self) -> None:
        payload = {"contributors": [c.to_dict() for c in self._contributors.values()]}
        index_path = self.directory / self.INDEX_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".contributors-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, index_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _persist_contributions(self) -> None:
        all_contributions = []
        for records in self._contributions.values():
            all_contributions.extend(records)
        payload = {"contributions": [r.to_dict() for r in all_contributions]}
        contrib_path = self.directory / self.CONTRIBUTIONS_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".contributions-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, contrib_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------------ contributors

    def register(
        self,
        name: str,
        email: str,
        role: str = ContributorRole.DEVELOPER,
        organization: str | None = None,
        affiliations: Iterable[str] = (),
        github_username: str | None = None,
        orcid: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        contributor_id: str | None = None,
    ) -> Contributor:
        """Register a new contributor."""
        with self._lock:
            if contributor_id is None:
                contributor_id = str(uuid.uuid4())
            if contributor_id in self._contributors:
                raise ValueError(f"contributor {contributor_id!r} already exists")
            # Check email uniqueness
            for c in self._contributors.values():
                if c.email == email:
                    raise ValueError(f"contributor with email {email!r} already exists")

            contributor = Contributor(
                id=contributor_id,
                name=name,
                email=email,
                role=role,
                organization=organization,
                affiliations=tuple(affiliations),
                github_username=github_username,
                orcid=orcid,
                metadata=dict(metadata or {}),
            )
            self._contributors[contributor_id] = contributor
            self._persist_contributors()
            logger.info("registered contributor %s (%s)", name, contributor_id)
            return contributor

    def get(self, contributor_id: str) -> Contributor | None:
        """Get a contributor by ID."""
        with self._lock:
            return self._contributors.get(contributor_id)

    def get_by_email(self, email: str) -> Contributor | None:
        """Get a contributor by email."""
        with self._lock:
            for c in self._contributors.values():
                if c.email == email:
                    return c
            return None

    def update(
        self,
        contributor_id: str,
        name: str | None = None,
        email: str | None = None,
        role: str | None = None,
        organization: str | None = None,
        affiliations: Iterable[str] | None = None,
        github_username: str | None = None,
        orcid: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Contributor:
        """Update a contributor's profile."""
        with self._lock:
            contributor = self._contributors.get(contributor_id)
            if contributor is None:
                raise KeyError(f"contributor {contributor_id!r} not found")

            # Check email uniqueness if changing
            if email is not None and email != contributor.email:
                for c in self._contributors.values():
                    if c.id != contributor_id and c.email == email:
                        raise ValueError(f"contributor with email {email!r} already exists")

            # Create updated contributor (dataclass is frozen-ish, recreate)
            updated = Contributor(
                id=contributor.id,
                name=name if name is not None else contributor.name,
                email=email if email is not None else contributor.email,
                role=role if role is not None else contributor.role,
                organization=organization if organization is not None else contributor.organization,
                affiliations=tuple(affiliations)
                if affiliations is not None
                else contributor.affiliations,
                github_username=github_username
                if github_username is not None
                else contributor.github_username,
                orcid=orcid if orcid is not None else contributor.orcid,
                created_at=contributor.created_at,
                updated_at=time.time(),
                metadata={**contributor.metadata, **(metadata or {})},
            )
            self._contributors[contributor_id] = updated
            self._persist_contributors()
            logger.info("updated contributor %s", contributor_id)
            return updated

    def delete(self, contributor_id: str) -> bool:
        """Delete a contributor (and their contribution records)."""
        with self._lock:
            if contributor_id not in self._contributors:
                return False
            del self._contributors[contributor_id]
            self._contributions.pop(contributor_id, None)
            self._persist_contributors()
            self._persist_contributions()
            logger.info("deleted contributor %s", contributor_id)
            return True

    def list_contributors(
        self,
        role: str | None = None,
        organization: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Contributor]:
        """List contributors with optional filtering."""
        with self._lock:
            contributors = list(self._contributors.values())
            if role:
                contributors = [c for c in contributors if c.role == role]
            if organization:
                contributors = [c for c in contributors if c.organization == organization]
            # Sort by created_at descending
            contributors.sort(key=lambda c: c.created_at, reverse=True)
            return contributors[offset : offset + limit]

    def count_contributors(self, role: str | None = None, organization: str | None = None) -> int:
        """Count contributors with optional filtering."""
        with self._lock:
            contributors = list(self._contributors.values())
            if role:
                contributors = [c for c in contributors if c.role == role]
            if organization:
                contributors = [c for c in contributors if c.organization == organization]
            return len(contributors)

    # ------------------------------------------------------------ contributions

    def add_contribution(
        self,
        contributor_id: str,
        contribution_type: str,
        target_id: str,
        description: str,
        impact_score: float = 0.0,
        verified: bool = False,
        verifier_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        contribution_id: str | None = None,
    ) -> ContributionRecord:
        """Record a contribution."""
        with self._lock:
            if contributor_id not in self._contributors:
                raise KeyError(f"contributor {contributor_id!r} not found")
            if contribution_id is None:
                contribution_id = str(uuid.uuid4())

            record = ContributionRecord(
                id=contribution_id,
                contributor_id=contributor_id,
                contribution_type=contribution_type,
                target_id=target_id,
                description=description,
                impact_score=impact_score,
                verified=verified,
                verifier_id=verifier_id,
                metadata=dict(metadata or {}),
            )
            self._contributions.setdefault(contributor_id, []).append(record)
            self._persist_contributions()
            logger.info("added contribution %s by %s", contribution_id, contributor_id)
            return record

    def get_contributions(
        self, contributor_id: str, contribution_type: str | None = None
    ) -> list[ContributionRecord]:
        """Get contributions for a contributor."""
        with self._lock:
            records = list(self._contributions.get(contributor_id, []))
            if contribution_type:
                records = [r for r in records if r.contribution_type == contribution_type]
            records.sort(key=lambda r: r.created_at, reverse=True)
            return records

    def get_contribution(self, contribution_id: str) -> ContributionRecord | None:
        """Get a specific contribution by ID."""
        with self._lock:
            for records in self._contributions.values():
                for r in records:
                    if r.id == contribution_id:
                        return r
            return None

    def verify_contribution(self, contribution_id: str, verifier_id: str) -> bool:
        """Mark a contribution as verified."""
        with self._lock:
            for records in self._contributions.values():
                for r in records:
                    if r.id == contribution_id:
                        r.verified = True
                        r.verifier_id = verifier_id
                        self._persist_contributions()
                        logger.info("verified contribution %s by %s", contribution_id, verifier_id)
                        return True
            return False

    def update_impact_score(self, contribution_id: str, impact_score: float) -> bool:
        """Update the impact score of a contribution."""
        with self._lock:
            for records in self._contributions.values():
                for r in records:
                    if r.id == contribution_id:
                        r.impact_score = impact_score
                        self._persist_contributions()
                        logger.info(
                            "updated impact score for %s to %.3f", contribution_id, impact_score
                        )
                        return True
            return False

    def list_all_contributions(
        self,
        contribution_type: str | None = None,
        verified: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContributionRecord]:
        """List all contributions with optional filtering."""
        with self._lock:
            all_records = []
            for records in self._contributions.values():
                all_records.extend(records)
            if contribution_type:
                all_records = [r for r in all_records if r.contribution_type == contribution_type]
            if verified is not None:
                all_records = [r for r in all_records if r.verified == verified]
            all_records.sort(key=lambda r: r.created_at, reverse=True)
            return all_records[offset : offset + limit]

    # ------------------------------------------------------------ export/import

    def export(self) -> dict[str, Any]:
        """Export all data for backup/portability."""
        with self._lock:
            return {
                "contributors": [c.to_dict() for c in self._contributors.values()],
                "contributions": [
                    r.to_dict() for records in self._contributions.values() for r in records
                ],
            }

    def import_data(self, data: Mapping[str, Any], overwrite: bool = False) -> int:
        """Import data from export format."""
        with self._lock:
            imported = 0
            for item in data.get("contributors", []):
                contributor = Contributor.from_dict(item)
                if contributor.id in self._contributors and not overwrite:
                    continue
                self._contributors[contributor.id] = contributor
                imported += 1
            for item in data.get("contributions", []):
                record = ContributionRecord.from_dict(item)
                self._contributions.setdefault(record.contributor_id, []).append(record)
                imported += 1
            self._persist_contributors()
            self._persist_contributions()
            logger.info("imported %d records", imported)
            return imported


__all__ = [
    "ContributionRecord",
    "Contributor",
    "ContributorRegistry",
    "ContributorRole",
]
