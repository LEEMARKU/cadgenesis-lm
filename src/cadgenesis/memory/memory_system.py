"""cadgenesis.memory.memory_system
=================================
Unified facade for the semantic memory layer.

Composes all eight domain pools, a cross-pool retriever, a router and a
pruner behind one API, and wires the pool capacities from ``MemoryConfig``.
This is the semantic counterpart of the torch
:class:`~cadgenesis.memory.memory_pools.LayerIntegratedMemorySystem`.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.config import MemoryConfig
from cadgenesis.memory.cad_memory import CADMemory
from cadgenesis.memory.engineering_memory import EngineeringMemory
from cadgenesis.memory.manufacturing_memory import ManufacturingMemory
from cadgenesis.memory.memory_common import MemoryStore
from cadgenesis.memory.memory_router import MemoryRouter, RoutingDecision
from cadgenesis.memory.persistence import MemoryPersistence
from cadgenesis.memory.project_memory import ProjectMemory
from cadgenesis.memory.pruning import MemoryPruner, PruningReport
from cadgenesis.memory.retrieval import MemoryRetrieval, RetrievalResult
from cadgenesis.memory.session_memory import SessionMemory
from cadgenesis.memory.simulation_memory import SimulationMemory
from cadgenesis.memory.user_memory import UserMemory
from cadgenesis.memory.working_memory import WorkingMemory

_DEFAULT_KEYWORDS: dict[str, set[str]] = {
    "working": {"context", "active", "current", "draft"},
    "session": {"session", "toolbar", "ui", "interaction"},
    "user": {"preference", "style", "user", "profile"},
    "project": {"project", "version", "milestone", "snapshot"},
    "cad": {"feature", "brep", "extrude", "fillet", "sketch", "part", "assembly"},
    "engineering": {"iso", "asme", "din", "standard", "tolerance", "material"},
    "manufacturing": {"machining", "tolerance", "tool", "draft", "mold", "3d-print"},
    "simulation": {"fea", "cfd", "stress", "strain", "mesh", "load", "safety"},
}


class MemorySystem:
    """All-in-one semantic memory facade."""

    POOLS = [
        "working",
        "session",
        "user",
        "project",
        "cad",
        "engineering",
        "manufacturing",
        "simulation",
    ]

    def __init__(
        self,
        capacities: dict[str, int] | None = None,
        domain_keywords: dict[str, set[str]] | None = None,
    ):
        capacities = capacities or {}
        self.working = WorkingMemory(capacity=capacities.get("working", 64))
        self.session = SessionMemory(capacity=capacities.get("session", 128))
        self.user = UserMemory(capacity=capacities.get("user", 256))
        self.project = ProjectMemory(capacity=capacities.get("project", 512))
        self.cad = CADMemory(capacity=capacities.get("cad", 1024))
        self.engineering = EngineeringMemory(capacity=capacities.get("engineering", 512))
        self.manufacturing = ManufacturingMemory(capacity=capacities.get("manufacturing", 512))
        self.simulation = SimulationMemory(capacity=capacities.get("simulation", 512))
        self._pools: dict[str, MemoryStore] = {
            "working": self.working,
            "session": self.session,
            "user": self.user,
            "project": self.project,
            "cad": self.cad,
            "engineering": self.engineering,
            "manufacturing": self.manufacturing,
            "simulation": self.simulation,
        }
        keywords = dict(_DEFAULT_KEYWORDS)
        if domain_keywords:
            keywords.update(domain_keywords)
        self.retriever = MemoryRetrieval(list(self._pools.values()))
        self.router = MemoryRouter(list(self._pools.values()), keywords)
        self.pruner = MemoryPruner()
        self.persistence = MemoryPersistence()

    # ------------------------------------------------------------- factories

    @classmethod
    def from_config(cls, memory_cfg: MemoryConfig) -> MemorySystem:
        """Build with slot counts from a ``MemoryConfig``."""
        capacities = {
            "working": memory_cfg.working_memory_slots,
            "session": memory_cfg.session_memory_slots,
            "user": memory_cfg.user_memory_slots,
            "project": memory_cfg.project_memory_slots,
            "cad": memory_cfg.cad_memory_slots,
            "engineering": memory_cfg.engineering_memory_slots,
            "manufacturing": memory_cfg.manufacturing_memory_slots,
            "simulation": memory_cfg.simulation_memory_slots,
        }
        return cls(capacities=capacities)

    # ----------------------------------------------------------------- pools

    def pool(self, name: str) -> MemoryStore:
        if name not in self._pools:
            raise KeyError(f"Unknown pool {name!r}; choose from {self.POOLS}")
        return self._pools[name]

    def register_store(
        self,
        store: MemoryStore,
        keywords: set[str] | None = None,
    ) -> None:
        """Add an extra store (e.g. the Pillar 6 long-term pool) at runtime.

        The default facade keeps the canonical eight pools; callers opt into a
        ninth (or custom) store by registering it.  The store is automatically
        wired into the retriever and the router.  Registering a name that
        already exists replaces the existing store.
        """
        name = store.name
        if not name:
            raise ValueError("cannot register an unnamed store")
        self._pools[name] = store
        if name not in self.POOLS:
            self.POOLS = [*self.POOLS, name]
        self.retriever.register(store)
        if keywords:
            self.router.register(store, keywords)
        else:
            self.router.register(store)

    @property
    def pools(self) -> dict[str, MemoryStore]:
        return dict(self._pools)

    @property
    def total_slots(self) -> int:
        return sum(len(pool) for pool in self._pools.values())

    # ----------------------------------------------------------------- write

    def remember(
        self,
        pool: str,
        key: str,
        content: Any,
        **kwargs: Any,
    ) -> Any:
        """Write into a named pool; returns the stored entry."""
        return self.pool(pool).add(key, content, **kwargs)

    def forget(self, pool: str, key: str) -> bool:
        return self.pool(pool).remove(key)

    # ----------------------------------------------------------------- read

    def recall(self, pool: str, key: str) -> Any:
        return self.pool(pool).get(key)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        pool_names: list[str] | None = None,
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k, pool_names=pool_names)

    def route(self, query: str) -> list[RoutingDecision]:
        return self.router.route(query)

    # ------------------------------------------------------------------- ops

    def prune(
        self,
        pool_names: list[str] | None = None,
        policy: str = "capacity",
        **params: Any,
    ) -> list[PruningReport]:
        names = pool_names or self.POOLS
        return self.pruner.prune_all([self.pool(name) for name in names], policy, **params)

    def save(self, directory: str) -> list:
        """Persist every pool to ``directory``. Returns written paths."""
        return self.persistence.save_many(list(self._pools.values()), directory)

    def load(self, directory: str, names: list[str] | None = None) -> None:
        """Load pools back from ``directory`` (replaces existing records)."""
        stores = self.persistence.load_many(directory, names)
        for name, store in stores.items():
            target = self._pools.get(name)
            if target is None:
                continue
            target.clear()
            for entry in store.values():
                target.add(
                    entry.key,
                    entry.content,
                    importance=entry.importance,
                    metadata=entry.metadata,
                )

    def summary(self) -> dict[str, Any]:
        return {
            "pools": {name: pool.summary() for name, pool in self._pools.items()},
            "total_slots": self.total_slots,
        }
