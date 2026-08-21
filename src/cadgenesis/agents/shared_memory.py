"""cadgenesis.agents.shared_memory
================================
Shared workspace for agent teams — a lock-guarded blackboard agents read from
and write to while collaborating on a design task.

Two tiers coexist:

* :class:`SharedMemory` — the legacy flat blackboard (unchanged).
* :class:`LayeredSharedMemory` — Pillar 5 tiered memory (working / session /
  project / global / agent) with TTL, capacity bounds, change notifications
  and an optional mirror into the semantic :class:`MemorySystem`.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class SharedMemory:
    """Thread-safe key/value blackboard shared by all agents."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ----------------------------------------------------------------- write

    def set(self, key: str, value: Any) -> None:
        """Write (or overwrite) a shared value."""
        with self._lock:
            self._data[key] = value

    def update(self, mapping: dict[str, Any]) -> None:
        """Merge several key/value pairs at once."""
        with self._lock:
            self._data.update(mapping)

    def remove(self, key: str) -> bool:
        """Delete a shared value; returns True when it existed."""
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # ----------------------------------------------------------------- read

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    @property
    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)

    def items(self) -> dict[str, Any]:
        """A shallow snapshot of the whole blackboard."""
        with self._lock:
            return dict(self._data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        return self.contains(key)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)


@dataclass
class MemoryRegion:
    """A named tier of the :class:`LayeredSharedMemory`."""

    name: str
    capacity: int = 1024
    ttl: float | None = None
    data: dict[str, tuple[Any, float]] = field(default_factory=dict)


class LayeredSharedMemory:
    """Tiered, TTL-aware shared memory for the agent fleet.

    Regions (``working``, ``session``, ``project``, ``global``, ``agent``)
    isolate concerns while the whole workspace stays readable.  Each region is
    capacity-bounded and may expire entries after ``ttl`` seconds.  Optionally
    mirrors writes into a semantic :class:`MemorySystem` pool for persistence.
    """

    DEFAULT_REGIONS = ("working", "session", "project", "global", "agent")

    def __init__(
        self,
        regions: tuple[str, ...] = DEFAULT_REGIONS,
        default_capacity: int = 1024,
        default_ttl: float | None = None,
        memory_system: Any = None,
        memory_pool: str = "agent",
    ) -> None:
        if default_capacity < 1:
            raise ValueError("default_capacity must be >= 1")
        self._regions: dict[str, MemoryRegion] = {
            name: MemoryRegion(name=name, capacity=default_capacity, ttl=default_ttl)
            for name in regions
        }
        self._memory_system = memory_system
        self._memory_pool = memory_pool
        self._listeners: dict[str, list[Callable[[str, str, Any], None]]] = {}
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- regions

    @property
    def regions(self) -> list[str]:
        with self._lock:
            return list(self._regions)

    def region_names(self) -> list[str]:
        return self.regions

    def _region(self, region: str) -> MemoryRegion:
        if region not in self._regions:
            raise KeyError(f"unknown memory region {region!r}")
        return self._regions[region]

    def exists_region(self, region: str) -> bool:
        return region in self._regions

    def add_region(self, name: str, capacity: int = 1024, ttl: float | None = None) -> None:
        if name in self._regions:
            raise ValueError(f"region {name!r} already exists")
        self._regions[name] = MemoryRegion(name=name, capacity=capacity, ttl=ttl)

    # ------------------------------------------------------------------ write

    def set(self, region: str, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            reg = self._region(region)
            reg.data[key] = (value, now)
            if len(reg.data) > reg.capacity:
                oldest = min(reg.data, key=lambda k: reg.data[k][1])
                del reg.data[oldest]
        self._notify(region, key, value)
        self._mirror(region, key, value)

    def update(self, region: str, mapping: dict[str, Any]) -> None:
        for key, value in mapping.items():
            self.set(region, key, value)

    def remove(self, region: str, key: str) -> bool:
        with self._lock:
            return self._region(region).data.pop(key, None) is not None

    def clear(self, region: str | None = None) -> None:
        with self._lock:
            if region is None:
                for reg in self._regions.values():
                    reg.data.clear()
            else:
                self._region(region).data.clear()

    # ------------------------------------------------------------------- read

    def get(self, region: str, key: str, default: Any = None) -> Any:
        now = time.time()
        with self._lock:
            reg = self._region(region)
            entry = reg.data.get(key)
            if entry is None:
                return default
            value, inserted = entry
            if reg.ttl is not None and now - inserted > reg.ttl:
                del reg.data[key]
                return default
            return value

    def peek(self, region: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._region(region).data.get(key, (default, 0))[0]

    def keys(self, region: str) -> list[str]:
        now = time.time()
        with self._lock:
            reg = self._region(region)
            return [k for k, (_, t) in reg.data.items() if reg.ttl is None or now - t <= reg.ttl]

    def snapshot(self, region: str | None = None) -> dict[str, dict[str, Any]]:
        with self._lock:
            if region is not None:
                return {region: {k: v[0] for k, v in self._region(region).data.items()}}
            return {
                name: {k: v[0] for k, v in reg.data.items()} for name, reg in self._regions.items()
            }

    def __contains__(self, key: str) -> bool:
        return any(key in reg.data for reg in self._regions.values())

    # --------------------------------------------------------- change events

    def on_change(self, region: str, handler: Callable[[str, str, Any], None]) -> None:
        self._listeners.setdefault(region, []).append(handler)

    def _notify(self, region: str, key: str, value: Any) -> None:
        for handler in self._listeners.get(region, []):
            with contextlib.suppress(Exception):
                handler(region, key, value)

    # ----------------------------------------------------- semantic mirror

    def _mirror(self, region: str, key: str, value: Any) -> None:
        if self._memory_system is None:
            return
        try:
            self._memory_system.remember(self._memory_pool, f"{region}:{key}", value)
        except KeyError:
            # Unknown pool (e.g. a not-yet-registered "agent" tier): fall back
            # to the semantic project pool so mirroring is best-effort.
            with contextlib.suppress(Exception):
                self._memory_system.remember("project", f"{region}:{key}", value)
        except Exception:
            pass

    def attach_memory(self, memory_system: Any, pool: str = "agent") -> None:
        self._memory_system = memory_system
        self._memory_pool = pool

    # -------------------------------------------------------- knowledge cache

    def cache_put(self, key: str, value: Any, ttl: float = 300.0) -> None:
        with self._lock:
            self._cache[key] = (value, time.time() + ttl)

    def cache_get(self, key: str, default: Any = None) -> Any:
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return default
            value, expires = entry
            if now > expires:
                del self._cache[key]
                return default
            return value

    def cache_clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ meta

    def usage(self) -> dict[str, int]:
        with self._lock:
            return {name: len(reg.data) for name, reg in self._regions.items()}
