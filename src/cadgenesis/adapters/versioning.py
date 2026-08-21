"""cadgenesis.adapters.versioning
=============================
Adapter versioning.

Semantic-version tags (``<adapter_id>@v<major>.<minor>.<patch>``) for
adapters, with a registry supporting resolve / latest / bump operations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class AdapterVersion:
    """Semantic version for an adapter snapshot."""

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if value < 0:
                raise ValueError(f"version component {name!r} must be >= 0, got {value}")

    @classmethod
    def from_string(cls, value: str) -> AdapterVersion:
        """Parse ``"v1.2.3"`` or ``"1.2.3"`` into an :class:`AdapterVersion`."""
        raw = value[1:] if value.startswith("v") else value
        parts = raw.split(".")
        if len(parts) != 3:
            raise ValueError(f"invalid version string {value!r}; expected MAJOR.MINOR.PATCH")
        try:
            major, minor, patch = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"invalid version string {value!r}") from exc
        return cls(major=major, minor=minor, patch=patch)

    def to_string(self) -> str:
        """Render as ``"vMAJOR.MINOR.PATCH"``."""
        return f"v{self.major}.{self.minor}.{self.patch}"

    def is_compatible(self, other: AdapterVersion, min_major: int = 1) -> bool:
        """True when both versions share a major line >= ``min_major``."""
        return self.major == other.major and self.major >= min_major


class AdapterVersionRegistry:
    """Registry mapping adapter ids to their ordered version history."""

    def __init__(self) -> None:
        self._versions: dict[str, list[AdapterVersion]] = {}

    def register(self, adapter_id: str, version: AdapterVersion | str) -> str:
        """Record ``version`` for ``adapter_id``; returns its unique tag."""
        parsed = (
            version if isinstance(version, AdapterVersion) else AdapterVersion.from_string(version)
        )
        self._versions.setdefault(adapter_id, []).append(parsed)
        return f"{adapter_id}@{parsed.to_string()}"

    def resolve(self, tag: str) -> AdapterVersion:
        """Resolve a tag of the form ``"<adapter_id>@vMAJOR.MINOR.PATCH"``."""
        adapter_id, _, version_str = tag.rpartition("@")
        if not adapter_id:
            raise ValueError(f"invalid tag {tag!r}; expected <adapter_id>@vMAJOR.MINOR.PATCH")
        version = AdapterVersion.from_string(version_str)
        if version not in self._versions.get(adapter_id, []):
            raise KeyError(f"version {tag!r} is not registered")
        return version

    def latest(self, adapter_id: str) -> AdapterVersion | None:
        """Highest registered version for ``adapter_id`` (None if none)."""
        versions = self._versions.get(adapter_id)
        return max(versions) if versions else None

    def bump(self, adapter_id: str, part: str = "patch") -> AdapterVersion:
        """Increment the ``major`` / ``minor`` / ``patch`` component and register it."""
        if part not in ("major", "minor", "patch"):
            raise ValueError(f"invalid part {part!r}; expected 'major', 'minor', or 'patch'")
        current = self.latest(adapter_id)
        if current is None:
            new_version = AdapterVersion(major=1, minor=0, patch=0)
        elif part == "major":
            new_version = AdapterVersion(current.major + 1, 0, 0)
        elif part == "minor":
            new_version = AdapterVersion(current.major, current.minor + 1, 0)
        else:
            new_version = AdapterVersion(current.major, current.minor, current.patch + 1)
        self.register(adapter_id, new_version)
        return new_version
