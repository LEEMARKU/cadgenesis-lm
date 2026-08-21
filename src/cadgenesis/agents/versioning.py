"""cadgenesis.agents.versioning
============================
Semantic versioning for agents and the agent platform.

Agents declare a ``version`` string (``major.minor.patch``).  The platform uses
compatibility rules on the **major** component when deciding whether a plugin
agent can be loaded alongside the core fleet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class AgentVersion:
    """A parsed ``major.minor.patch`` version."""

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("version components must be non-negative")

    @classmethod
    def parse(cls, text: str) -> AgentVersion:
        """Parse ``"1.2.3"`` into an :class:`AgentVersion`.

        Raises ``ValueError`` for anything that is not ``major.minor.patch``.
        """
        if not _VERSION_RE.match(text or ""):
            raise ValueError(f"invalid agent version {text!r}: expected 'major.minor.patch'")
        major, minor, patch = (int(part) for part in text.split("."))
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: AgentVersion) -> bool:
        """Backward compatible when the major version matches."""
        return self.major == other.major

    def __lt__(self, other: AgentVersion) -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: AgentVersion) -> bool:
        return (self.major, self.minor, self.patch) <= (
            other.major,
            other.minor,
            other.patch,
        )


def parse_version(text: str) -> AgentVersion:
    """Convenience wrapper around :meth:`AgentVersion.parse`."""
    return AgentVersion.parse(text)


def validate_version(text: str) -> bool:
    """True when ``text`` is a valid ``major.minor.patch`` version."""
    return bool(_VERSION_RE.match(text or ""))
