"""
cadgenesis.collaboration.dataset_marketplace
============================================
Dataset Marketplace for the CADGenesis-LM Collaborative Research Economy.

Supports publication, versioning, reviews, and access control for datasets.
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

from cadgenesis.collaboration.contributors import ContributorRegistry
from cadgenesis.collaboration.reputation import ReputationEngine
from cadgenesis.research.datasets import DatasetRegistry, DatasetVersion

logger = logging.getLogger("cadgenesis.collaboration.dataset_marketplace")


class DatasetVisibility:
    """Dataset visibility levels."""

    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class DatasetStatus:
    """Dataset publication status."""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class DatasetMetadata:
    """Metadata for a published dataset."""

    id: str
    name: str
    description: str
    owner_id: str  # contributor_id
    organization: str | None = None
    visibility: str = DatasetVisibility.PRIVATE
    status: str = DatasetStatus.DRAFT
    tags: tuple[str, ...] = ()
    license: str = "MIT"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    published_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetMetadata:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            owner_id=str(data["owner_id"]),
            organization=data.get("organization"),
            visibility=str(data.get("visibility", DatasetVisibility.PRIVATE)),
            status=str(data.get("status", DatasetStatus.DRAFT)),
            tags=tuple(data.get("tags", [])),
            license=str(data.get("license", "MIT")),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            published_at=data.get("published_at"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DatasetReview:
    """A review of a dataset."""

    id: str
    dataset_id: str
    reviewer_id: str
    rating: int  # 1-5
    comment: str
    criteria_scores: dict[str, int] = field(
        default_factory=dict
    )  # e.g., quality, documentation, usability
    verified_purchase: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetReview:
        return cls(
            id=str(data["id"]),
            dataset_id=str(data["dataset_id"]),
            reviewer_id=str(data["reviewer_id"]),
            rating=int(data["rating"]),
            comment=str(data["comment"]),
            criteria_scores=dict(data.get("criteria_scores", {})),
            verified_purchase=bool(data.get("verified_purchase", False)),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


@dataclass
class DatasetAccess:
    """Access grant for a dataset."""

    id: str
    dataset_id: str
    grantee_id: str  # contributor_id or organization
    access_level: str = "read"  # read, write, admin
    granted_by: str = ""
    expires_at: float | None = None
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetAccess:
        return cls(
            id=str(data["id"]),
            dataset_id=str(data["dataset_id"]),
            grantee_id=str(data["grantee_id"]),
            access_level=str(data.get("access_level", "read")),
            granted_by=str(data.get("granted_by", "")),
            expires_at=data.get("expires_at"),
            created_at=float(data.get("created_at", 0.0)),
        )


class DatasetMarketplace:
    """
    Marketplace for publishing, discovering, and accessing datasets.

    Integrates with:
    - DatasetRegistry for versioning and storage
    - ContributorRegistry for ownership and contributions
    - ReputationEngine for reviewer credibility
    - ComputeCreditLedger for storage costs
    """

    METADATA_FILE = "dataset_metadata.json"
    REVIEWS_FILE = "dataset_reviews.json"
    ACCESS_FILE = "dataset_access.json"

    def __init__(
        self,
        directory: str | os.PathLike[str],
        dataset_registry: DatasetRegistry,
        contributor_registry: ContributorRegistry,
        reputation_engine: ReputationEngine | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset_registry = dataset_registry
        self.contributor_registry = contributor_registry
        self.reputation_engine = reputation_engine
        self._lock = threading.RLock()
        self._metadata: dict[str, DatasetMetadata] = {}
        self._reviews: dict[str, list[DatasetReview]] = {}  # dataset_id -> list
        self._access: dict[str, list[DatasetAccess]] = {}  # dataset_id -> list
        self._load()

    def _load(self) -> None:
        # Load metadata
        meta_path = self.directory / self.METADATA_FILE
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                for item in data.get("datasets", []):
                    meta = DatasetMetadata.from_dict(item)
                    self._metadata[meta.id] = meta
            except (ValueError, OSError) as exc:
                logger.warning("dataset metadata unreadable: %s", exc)

        # Load reviews
        reviews_path = self.directory / self.REVIEWS_FILE
        if reviews_path.exists():
            try:
                data = json.loads(reviews_path.read_text(encoding="utf-8"))
                for item in data.get("reviews", []):
                    review = DatasetReview.from_dict(item)
                    self._reviews.setdefault(review.dataset_id, []).append(review)
            except (ValueError, OSError) as exc:
                logger.warning("dataset reviews unreadable: %s", exc)

        # Load access
        access_path = self.directory / self.ACCESS_FILE
        if access_path.exists():
            try:
                data = json.loads(access_path.read_text(encoding="utf-8"))
                for item in data.get("access", []):
                    access = DatasetAccess.from_dict(item)
                    self._access.setdefault(access.dataset_id, []).append(access)
            except (ValueError, OSError) as exc:
                logger.warning("dataset access unreadable: %s", exc)

    def _persist_metadata(self) -> None:
        payload = {"datasets": [m.to_dict() for m in self._metadata.values()]}
        path = self.directory / self.METADATA_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".metadata-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _persist_reviews(self) -> None:
        all_reviews = []
        for reviews in self._reviews.values():
            all_reviews.extend(reviews)
        payload = {"reviews": [r.to_dict() for r in all_reviews]}
        path = self.directory / self.REVIEWS_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".reviews-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _persist_access(self) -> None:
        all_access = []
        for access_list in self._access.values():
            all_access.extend(access_list)
        payload = {"access": [a.to_dict() for a in all_access]}
        path = self.directory / self.ACCESS_FILE
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".access-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------------ publish

    def publish_dataset(
        self,
        name: str,
        description: str,
        owner_id: str,
        source_path: str | os.PathLike[str],
        version: str | None = None,
        organization: str | None = None,
        visibility: str = DatasetVisibility.PRIVATE,
        tags: Iterable[str] = (),
        license: str = "MIT",
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[DatasetMetadata, DatasetVersion]:
        """Publish a new dataset to the marketplace."""
        with self._lock:
            # Verify owner
            owner = self.contributor_registry.get(owner_id)
            if owner is None:
                raise KeyError(f"contributor {owner_id!r} not found")

            # Snapshot to dataset registry
            dataset_version = self.dataset_registry.snapshot(
                name=name,
                source=source_path,
                version=version,
                metadata=dict(metadata or {}),
            )

            # Create marketplace metadata
            dataset_id = str(uuid.uuid4())
            now = time.time()
            meta = DatasetMetadata(
                id=dataset_id,
                name=name,
                description=description,
                owner_id=owner_id,
                organization=organization,
                visibility=visibility,
                status=DatasetStatus.PUBLISHED
                if visibility != DatasetVisibility.PRIVATE
                else DatasetStatus.DRAFT,
                tags=tuple(tags),
                license=license,
                created_at=now,
                updated_at=now,
                published_at=now if visibility != DatasetVisibility.PRIVATE else None,
                metadata=dict(metadata or {}),
            )

            self._metadata[dataset_id] = meta
            self._persist_metadata()

            # Record contribution
            self.contributor_registry.add_contribution(
                contributor_id=owner_id,
                contribution_type="dataset",
                target_id=dataset_id,
                description=f"Published dataset {name} v{dataset_version.version}",
                metadata={"dataset_version": dataset_version.version},
            )

            logger.info("published dataset %s (%s) by %s", name, dataset_id, owner_id)
            return meta, dataset_version

    def update_dataset(
        self,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
        status: str | None = None,
        tags: Iterable[str] | None = None,
        license: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetMetadata:
        """Update dataset metadata."""
        with self._lock:
            meta = self._metadata.get(dataset_id)
            if meta is None:
                raise KeyError(f"dataset {dataset_id!r} not found")

            updated = DatasetMetadata(
                id=meta.id,
                name=name if name is not None else meta.name,
                description=description if description is not None else meta.description,
                owner_id=meta.owner_id,
                organization=meta.organization,
                visibility=visibility if visibility is not None else meta.visibility,
                status=status if status is not None else meta.status,
                tags=tuple(tags) if tags is not None else meta.tags,
                license=license if license is not None else meta.license,
                created_at=meta.created_at,
                updated_at=time.time(),
                published_at=meta.published_at,
                metadata={**meta.metadata, **(metadata or {})},
            )

            # Handle status transitions
            if status == DatasetStatus.PUBLISHED and meta.status != DatasetStatus.PUBLISHED:
                updated.published_at = time.time()

            self._metadata[dataset_id] = updated
            self._persist_metadata()
            return updated

    def add_version(
        self,
        dataset_id: str,
        source_path: str | os.PathLike[str],
        version: str | None = None,
        parent_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        """Add a new version to an existing dataset."""
        with self._lock:
            meta = self._metadata.get(dataset_id)
            if meta is None:
                raise KeyError(f"dataset {dataset_id!r} not found")

            # Snapshot new version
            dataset_version = self.dataset_registry.snapshot(
                name=meta.name,
                source=source_path,
                version=version,
                parent=parent_version,
                metadata=dict(metadata or {}),
            )

            # Update metadata timestamp
            meta.updated_at = time.time()
            self._persist_metadata()

            logger.info("added version %s to dataset %s", dataset_version.version, dataset_id)
            return dataset_version

    # ------------------------------------------------------------ access control

    def grant_access(
        self,
        dataset_id: str,
        grantee_id: str,
        access_level: str = "read",
        granted_by: str = "",
        expires_in_days: int | None = None,
    ) -> DatasetAccess:
        """Grant access to a dataset."""
        with self._lock:
            if dataset_id not in self._metadata:
                raise KeyError(f"dataset {dataset_id!r} not found")

            expires_at = None
            if expires_in_days is not None:
                expires_at = time.time() + (expires_in_days * 86400)

            access = DatasetAccess(
                id=str(uuid.uuid4()),
                dataset_id=dataset_id,
                grantee_id=grantee_id,
                access_level=access_level,
                granted_by=granted_by,
                expires_at=expires_at,
            )

            self._access.setdefault(dataset_id, []).append(access)
            self._persist_access()
            logger.info(
                "granted %s access to dataset %s for %s", access_level, dataset_id, grantee_id
            )
            return access

    def revoke_access(self, dataset_id: str, access_id: str) -> bool:
        """Revoke an access grant."""
        with self._lock:
            access_list = self._access.get(dataset_id, [])
            for i, access in enumerate(access_list):
                if access.id == access_id:
                    access_list.pop(i)
                    self._persist_access()
                    return True
            return False

    def check_access(
        self, dataset_id: str, contributor_id: str, required_level: str = "read"
    ) -> bool:
        """Check if a contributor has access to a dataset."""
        with self._lock:
            meta = self._metadata.get(dataset_id)
            if meta is None:
                return False

            # Owner always has access
            if meta.owner_id == contributor_id:
                return True

            # Public datasets are readable by all
            if meta.visibility == DatasetVisibility.PUBLIC and required_level == "read":
                return True

            # Check organization access
            if meta.visibility == DatasetVisibility.ORGANIZATION:
                grantee = self.contributor_registry.get(contributor_id)
                if grantee and grantee.organization == meta.organization:
                    return True

            # Check explicit grants
            access_list = self._access.get(dataset_id, [])
            levels = {"read": 1, "write": 2, "admin": 3}
            required = levels.get(required_level, 1)
            for access in access_list:
                if access.grantee_id == contributor_id and not access.is_expired():
                    granted = levels.get(access.access_level, 1)
                    if granted >= required:
                        return True

            return False

    def get_access_list(self, dataset_id: str) -> list[DatasetAccess]:
        """Get all access grants for a dataset."""
        with self._lock:
            return list(self._access.get(dataset_id, []))

    # ------------------------------------------------------------ reviews

    def add_review(
        self,
        dataset_id: str,
        reviewer_id: str,
        rating: int,
        comment: str,
        criteria_scores: Mapping[str, int] | None = None,
        verified_purchase: bool = False,
    ) -> DatasetReview:
        """Add a review for a dataset."""
        with self._lock:
            if dataset_id not in self._metadata:
                raise KeyError(f"dataset {dataset_id!r} not found")
            if not 1 <= rating <= 5:
                raise ValueError("rating must be between 1 and 5")

            reviewer = self.contributor_registry.get(reviewer_id)
            if reviewer is None:
                raise KeyError(f"reviewer {reviewer_id!r} not found")

            # Check if reviewer has access
            if not self.check_access(dataset_id, reviewer_id, "read"):
                raise PermissionError("reviewer does not have access to this dataset")

            review = DatasetReview(
                id=str(uuid.uuid4()),
                dataset_id=dataset_id,
                reviewer_id=reviewer_id,
                rating=rating,
                comment=comment,
                criteria_scores=dict(criteria_scores or {}),
                verified_purchase=verified_purchase,
            )

            self._reviews.setdefault(dataset_id, []).append(review)
            self._persist_reviews()

            # Record contribution for reviewer
            self.contributor_registry.add_contribution(
                contributor_id=reviewer_id,
                contribution_type="review",
                target_id=dataset_id,
                description=f"Reviewed dataset {self._metadata[dataset_id].name}",
                metadata={"rating": rating},
            )

            logger.info("added review %s for dataset %s by %s", review.id, dataset_id, reviewer_id)
            return review

    def get_reviews(self, dataset_id: str) -> list[DatasetReview]:
        """Get all reviews for a dataset."""
        with self._lock:
            return list(self._reviews.get(dataset_id, []))

    def get_average_rating(self, dataset_id: str) -> float | None:
        """Get average rating for a dataset."""
        reviews = self.get_reviews(dataset_id)
        if not reviews:
            return None
        return sum(r.rating for r in reviews) / len(reviews)

    # ------------------------------------------------------------ discovery

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        """Get dataset metadata by ID."""
        with self._lock:
            return self._metadata.get(dataset_id)

    def list_datasets(
        self,
        owner_id: str | None = None,
        organization: str | None = None,
        visibility: str | None = None,
        status: str | None = None,
        tags: Iterable[str] | None = None,
        contributor_id: str | None = None,  # Filter by access
        limit: int = 50,
        offset: int = 0,
    ) -> list[DatasetMetadata]:
        """List datasets with filtering."""
        with self._lock:
            datasets = list(self._metadata.values())

            if owner_id:
                datasets = [d for d in datasets if d.owner_id == owner_id]
            if organization:
                datasets = [d for d in datasets if d.organization == organization]
            if visibility:
                datasets = [d for d in datasets if d.visibility == visibility]
            if status:
                datasets = [d for d in datasets if d.status == status]
            if tags:
                tag_set = set(tags)
                datasets = [d for d in datasets if tag_set.intersection(d.tags)]
            if contributor_id:
                datasets = [d for d in datasets if self.check_access(d.id, contributor_id, "read")]

            # Sort by updated_at descending
            datasets.sort(key=lambda d: d.updated_at, reverse=True)
            return datasets[offset : offset + limit]

    def search_datasets(
        self, query: str, contributor_id: str | None = None, limit: int = 20
    ) -> list[DatasetMetadata]:
        """Search datasets by name/description/tags."""
        query_lower = query.lower()
        datasets = self.list_datasets(contributor_id=contributor_id, limit=1000)
        results = []
        for d in datasets:
            if (
                query_lower in d.name.lower()
                or query_lower in d.description.lower()
                or any(query_lower in tag.lower() for tag in d.tags)
            ):
                results.append(d)
                if len(results) >= limit:
                    break
        return results

    # ------------------------------------------------------------ versions

    def get_versions(self, dataset_id: str) -> list[DatasetVersion]:
        """Get all versions of a dataset."""
        meta = self.get_dataset(dataset_id)
        if meta is None:
            return []
        return self.dataset_registry.list_versions(meta.name)

    def get_version(self, dataset_id: str, version: str) -> DatasetVersion | None:
        """Get a specific version of a dataset."""
        meta = self.get_dataset(dataset_id)
        if meta is None:
            return None
        return self.dataset_registry.get(meta.name, version=version)

    def verify_dataset(self, dataset_id: str, version: str | None = None) -> bool:
        """Verify dataset integrity."""
        meta = self.get_dataset(dataset_id)
        if meta is None:
            return False
        target_version = version
        if not target_version:
            latest = self.dataset_registry.get(meta.name)
            assert latest is not None
            target_version = latest.version
        return self.dataset_registry.verify(meta.name, target_version)


__all__ = [
    "DatasetAccess",
    "DatasetMarketplace",
    "DatasetMetadata",
    "DatasetReview",
    "DatasetStatus",
    "DatasetVisibility",
]
