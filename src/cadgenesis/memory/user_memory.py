"""cadgenesis.memory.user_memory
===============================
User memory pool — persistent user preferences and design style.  Used to
personalise generation (materials, tolerances, file formats, naming habits).
"""

from __future__ import annotations

from typing import Any

from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore


class UserMemory(MemoryStore):
    """Persistent user preference store (default 256 records)."""

    def __init__(
        self,
        capacity: int = 256,
        default_importance: float = 1.0,
    ):
        super().__init__(
            name="user",
            capacity=capacity,
            default_importance=default_importance,
        )

    def set_preference(self, name: str, value: Any) -> MemoryEntry:
        """Store a named user preference."""
        return self.add(f"pref:{name}", value, metadata={"kind": "preference"})

    def get_preference(self, name: str, default: Any = None) -> Any:
        entry = self.peek(f"pref:{name}")
        if entry is None:
            return default
        entry.touch()
        return entry.content

    def record_style(self, style: dict[str, Any]) -> MemoryEntry:
        """Record an observed design-style fingerprint."""
        return self.add(
            f"style:{style.get('name', 'default')}",
            style,
            metadata={"kind": "style"},
        )

    def style(self, name: str = "default") -> dict[str, Any] | None:
        entry = self.peek(f"style:{name}")
        return entry.content if entry is not None else None

    def preferences(self) -> dict[str, Any]:
        return {
            entry.key[len("pref:") :]: entry.content
            for entry in self.entries()
            if entry.metadata.get("kind") == "preference"
        }
