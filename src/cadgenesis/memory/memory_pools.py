"""
cadgenesis.memory.memory_pools
=============================
Layer-Integrated Memory Pools for CADGenesis-LM v2.0.

The system exposes **8 semantically distinct memory pools**:

    1. working       — short-term context buffer for the active generation
    2. session       — current design session context
    3. project       — persistent project / design-state memory
    4. user          — user preferences & design style
    5. cad           — historical feature trees & B-Rep topologies
    6. engineering   — ISO / ASME / DIN standards & design guidelines
    7. manufacturing — machining & process limits (DFM)
    8. simulation    — past FEA/CFD results & safety factors

Every transformer layer accesses these pools through *memory attention* and
*refines* the working-memory region after each layer (see ``refine``), which
is what makes the memory genuinely *layer-integrated* rather than a single
pre-computed context vector.

Architecture
------------
::

    LayerIntegratedMemorySystem
    ├── working_memory       : MemoryPool(16 slots)
    ├── session_memory       : MemoryPool(32 slots)
    ├── project_memory       : MemoryPool(64 slots)
    ├── user_memory          : MemoryPool(16 slots)
    ├── cad_memory           : MemoryPool(64 slots)
    ├── engineering_memory   : MemoryPool(32 slots)
    ├── manufacturing_memory : MemoryPool(32 slots)
    ├── simulation_memory    : MemoryPool(32 slots)
    ├── get_combined_memory_bank() → (B, 288, C)
    ├── retrieve()           → top-k RAG retrieval over all pools
    └── refine()             → per-layer differentiable write-back into working pool

Complexity
----------
    Combined bank:     O(total_slots · d) memory
    Top-k retrieve:    O(B · T · total_slots · d)   (matmul over slots)
    Per-layer refine:  O(B · C)  (mean-pool + gate-blend into 1 slot group)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.config import MemoryConfig


class MemoryPool(nn.Module):
    """Base class for a memory pool with slot-based storage and retrieval."""

    def __init__(self, name: str, num_slots: int, d_model: int):
        super().__init__()
        self.name = name
        self.num_slots = num_slots
        self.d_model = d_model
        self.memory_slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

    def get_memory(self, batch_size: int = 1) -> torch.Tensor:
        """Returns memory tensor of shape (batch_size, num_slots, d_model)."""
        return self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)

    def write_memory(self, indices: torch.Tensor, new_states: torch.Tensor):
        """Updates specific slots in the memory pool (in place, no grad)."""
        with torch.no_grad():
            self.memory_slots[indices] = new_states.detach()

    def write_memory_all(self, new_states: torch.Tensor):
        """Overwrite the whole pool (in place, no grad)."""
        with torch.no_grad():
            self.memory_slots.copy_(new_states.detach())

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cosine-similarity retrieval from this pool.

        query: (B, T, C) or (B, C) → returns:
            values: (B, T or 1, min(top_k, num_slots), C)
            scores: (B, T or 1, min(top_k, num_slots))
        """
        squeeze_query = query.dim() == 2
        if squeeze_query:
            query = query.unsqueeze(1)  # (B, 1, C)
        B, T, _C = query.shape
        qn = F.normalize(query, dim=-1)
        sn = F.normalize(self.memory_slots.unsqueeze(0), dim=-1)
        scores = torch.matmul(qn, sn.transpose(-2, -1))  # (B, T, M)
        k = min(top_k, self.num_slots)
        top_scores, top_idx = scores.topk(k, dim=-1)
        vals = self._gather_slots(top_idx, B, T, k)
        if squeeze_query:
            return vals[:, 0], top_scores[:, 0]
        return vals, top_scores

    def _gather_slots(self, top_idx: torch.Tensor, B: int, T: int, k: int) -> torch.Tensor:
        """Gather slot vectors given (B, T, k) indices → (B, T, k, C)."""
        slots = self.memory_slots.unsqueeze(0).expand(B, -1, -1)  # (B, M, C)
        return slots.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model))


class LayerIntegratedMemorySystem(nn.Module):
    """
    Unified 8-Tier Layer-Integrated Memory System for CADGenesis-LM v2.0.

    The default slot counts total **288** slots (16+32+64+16+64+32+32+32),
    matching the original CADGenesis v2 compact layout.  Full-scale counts are
    available through :meth:`from_config` using a ``MemoryConfig``.
    """

    POOLS: list[str] = [
        "working",
        "session",
        "project",
        "user",
        "cad",
        "engineering",
        "manufacturing",
        "simulation",
    ]

    DEFAULT_SLOTS: dict[str, int] = {
        "working": 16,
        "session": 32,
        "project": 64,
        "user": 16,
        "cad": 64,
        "engineering": 32,
        "manufacturing": 32,
        "simulation": 32,
    }

    def __init__(
        self,
        d_model: int = 1024,
        slots: dict[str, int] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        slots = dict(self.DEFAULT_SLOTS if slots is None else slots)
        if set(slots.keys()) != set(self.POOLS):
            raise ValueError(
                f"Memory slots must exactly cover pools {self.POOLS}; got {sorted(slots)}"
            )

        self._slots: dict[str, int] = {}
        for name in self.POOLS:
            num_slots = slots[name]
            self._slots[name] = num_slots
            setattr(self, f"{name}_memory", MemoryPool(name, num_slots, d_model))

        self.retrieval_proj = nn.Linear(d_model, d_model)
        # Per-layer differentiable refinement (write-back into working pool).
        self.refinement_proj = nn.Linear(d_model, d_model)
        self.refinement_gate = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------- factories

    @classmethod
    def from_config(cls, memory_cfg: MemoryConfig, d_model: int) -> LayerIntegratedMemorySystem:
        """Build from a MemoryConfig (full-scale pool counts)."""
        slots = {
            "working": memory_cfg.working_memory_slots,
            "session": memory_cfg.session_memory_slots,
            "project": memory_cfg.project_memory_slots,
            "user": memory_cfg.user_memory_slots,
            "cad": memory_cfg.cad_memory_slots,
            "engineering": memory_cfg.engineering_memory_slots,
            "manufacturing": memory_cfg.manufacturing_memory_slots,
            "simulation": memory_cfg.simulation_memory_slots,
        }
        return cls(d_model=d_model, slots=slots)

    # --------------------------------------------------------------- access

    def get_pool(self, name: str) -> MemoryPool:
        if name not in self.POOLS:
            raise KeyError(f"Unknown memory pool {name!r}; choose from {self.POOLS}")
        return getattr(self, f"{name}_memory")

    def memory_slot_offsets(self) -> dict[str, int]:
        """Start index of each pool inside the combined memory bank."""
        offsets: dict[str, int] = {}
        cursor = 0
        for name in self.POOLS:
            offsets[name] = cursor
            cursor += self._slots[name]
        return offsets

    @property
    def total_slots(self) -> int:
        return sum(self._slots.values())

    def get_combined_memory_bank(self, batch_size: int = 1) -> torch.Tensor:
        """
        Concatenates all 8 memory pools into a single unified memory bank tensor
        of shape (batch_size, total_slots, d_model).
        """
        pools = [self.get_pool(name).get_memory(batch_size) for name in self.POOLS]
        combined = torch.cat(pools, dim=1)  # (B, total_slots, C)
        return self.retrieval_proj(combined)

    def get_pool_bank(self, name: str, batch_size: int = 1) -> torch.Tensor:
        """Raw (unprojected) memory of a single pool: (B, slots, C)."""
        return self.get_pool(name).get_memory(batch_size)

    # ------------------------------------------------------------ write API

    def write_memory(self, pool_name: str, indices: torch.Tensor, new_states: torch.Tensor):
        """Write into a named pool (no grad)."""
        self.get_pool(pool_name).write_memory(indices, new_states)

    def write_memory_all(self, pool_name: str, new_states: torch.Tensor):
        """Overwrite a named pool (no grad)."""
        self.get_pool(pool_name).write_memory_all(new_states)

    # ------------------------------------------------------------- retrieval

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 8,
        pool_names: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Cross-pool top-k retrieval (RAG-style).

        query: (B, C) or (B, T, C)
        pool_names: optional subset of pools to search (default: all).

        Returns
        -------
        (values (B, T, k, C), scores (B, T, k), source_pool_names (k,))
        """
        names = pool_names or self.POOLS
        squeeze_query = query.dim() == 2
        if squeeze_query:
            query = query.unsqueeze(1)

        candidates: list[tuple[torch.Tensor, str]] = []
        for name in names:
            pool = self.get_pool(name)
            qn = F.normalize(query, dim=-1)
            sn = F.normalize(pool.memory_slots.unsqueeze(0), dim=-1)
            scores = torch.matmul(qn, sn.transpose(-2, -1))  # (B, T, M_pool)
            candidates.append((scores, name))

        merged_scores = torch.cat([s[0] for s in candidates], dim=-1)  # (B, T, total)
        k = min(top_k, merged_scores.shape[-1])
        top_scores, top_idx = merged_scores.topk(k, dim=-1)

        # Map flat index → (pool, intra-pool index)
        values = torch.zeros(*top_scores.shape, self.d_model, device=query.device)
        sources: list[str] = []
        offset = 0
        for name in names:
            m = self._slots[name]
            mask = (top_idx >= offset) & (top_idx < offset + m)
            if mask.any():
                local = top_idx - offset
                slots = self.get_pool(name).memory_slots  # (M, C)
                gathered = slots[local.clamp(min=0, max=m - 1)]  # (B, T, k, C)
                values = torch.where(mask.unsqueeze(-1), gathered, values)
            offset += m

        # Map the *first* index of each k position to a source pool name.
        flat_positions = top_idx[:, 0, :].reshape(-1)
        source_map: dict[int, str] = {}
        off = 0
        for name in names:
            for idx in range(off, off + self._slots[name]):
                source_map[idx] = name
            off += self._slots[name]
        sources = [source_map[int(p)] for p in flat_positions.tolist()]

        if squeeze_query:
            return values[:, 0], top_scores[:, 0], sources
        return values, top_scores, sources

    # ------------------------------------------------------ layer refinement

    def refine(self, memory_bank: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Differentiable per-layer memory refinement.

        Mean-pools ``hidden_states``, projects it into the embedding space and
        blends it into the *working* memory region of the bank with a learned
        gate.  Returns a new bank tensor (no in-place mutation of the graph),
        so deeper layers observe an evolving memory context.

        memory_bank:   (B, total_slots, C)
        hidden_states: (B, T, C)
        Returns:       (B, total_slots, C)
        """
        pooled = hidden_states.mean(dim=1)  # (B, C)
        candidate = self.refinement_proj(pooled).unsqueeze(1)  # (B, 1, C)
        gate = torch.sigmoid(self.refinement_gate(pooled)).unsqueeze(1)  # (B, 1, C)
        ws = self._slots["working"]
        working = memory_bank[:, :ws]
        updated = gate * candidate + (1.0 - gate) * working
        return torch.cat([updated, memory_bank[:, ws:]], dim=1)
