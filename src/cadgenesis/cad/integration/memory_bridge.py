"""cadgenesis.cad.integration.memory_bridge
=========================================
Store and retrieve CAD designs in the existing semantic ``CADMemory`` pool.

Designs are stored as serialisable dicts keyed by name; retrieval is by
keyword search over the stored content plus optional object-kind filtering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cadgenesis.memory.cad_memory import CADMemory


class CADMemoryBridge:
    """Bridges :class:`CADMemory` with the new CAD package objects."""

    def __init__(self, memory: CADMemory | None = None) -> None:
        self.memory = memory or CADMemory(capacity=1024)

    # -- writing ----------------------------------------------------------------

    def store_design(
        self,
        name: str,
        design: dict[str, Any],
        kind: str = "part",
        importance: float | None = None,
        overwrite: bool = True,
    ) -> Any:
        """Store a serialisable design dict under ``name``.

        Returns the stored :class:`~cadgenesis.memory.memory_common.MemoryEntry`.
        """
        key = _design_key(name)
        if not overwrite and self.memory.get(key) is not None:
            raise KeyError(f"design {name!r} already stored")
        return self.memory.add(
            key,
            _json_safe(design),
            importance=importance,
            metadata={"kind": "cad_design", "object_kind": kind, "name": name},
        )

    def store_brep(self, name: str, brep: dict[str, Any], kind: str = "part") -> Any:
        return self.memory.remember_brep(_design_key(name), _json_safe(brep), kind=kind)

    def store_feature_tree(
        self, name: str, feature_tree: list[dict[str, Any]], kind: str = "part"
    ) -> Any:
        return self.memory.remember_feature_tree(_design_key(name), feature_tree, kind=kind)

    # -- reading ----------------------------------------------------------------

    def recall(self, query: str, top_k: int = 8, kind: str | None = None) -> list[Any]:
        """Search stored designs; returns list of ``SearchResult``."""
        hits = self.memory.recall(query, top_k=top_k)
        if kind is not None:
            hits = [h for h in hits if h.entry.metadata.get("object_kind") == kind]
        return hits

    def get_design(self, name: str) -> dict[str, Any] | None:
        entry = self.memory.get(_design_key(name))
        return entry.content if entry is not None else None

    def list_designs(self, kind: str | None = None) -> list[str]:
        entries = self.memory.by_kind(kind) if kind else self.memory.entries()
        return [e.metadata.get("name", e.key) for e in entries]

    def size(self) -> int:
        return len(self.memory.entries())


def _design_key(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"design:{name}:{digest}"


def _json_safe(obj: Any) -> Any:
    """Return a JSON-serialisable copy (round-trips floats, tuples, Vecs)."""
    return json.loads(json.dumps(obj, default=_default_json))


def _default_json(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict, list)):
        return list(value)
    return repr(value)


__all__ = ["CADMemoryBridge"]
