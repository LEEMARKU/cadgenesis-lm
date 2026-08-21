"""
cadgenesis.training.fsdp
========================
PyTorch FullyShardedDataParallel (FSDP) helpers for CADGenesis-LM.

``wrap_fsdp`` shards model parameters across ranks with configurable
sharding strategy, mixed precision and CPU-offload; ``fsdp_sharded_ckpt``
handles the optimizers' sharded state for resume.  All functionality
degrades gracefully when torch.distributed is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

try:
    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint import (  # type: ignore[attr-defined]  # torch stubs omit FilePlanarPlacement
        FilePlanarPlacement,
    )
    from torch.distributed.checkpoint import (
        load as dcp_load,
    )
    from torch.distributed.checkpoint import (
        save as dcp_save,
    )
    from torch.distributed.checkpoint.default_planner import (
        DefaultLoadPlanner,
        DefaultSavePlanner,
    )
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp.api import CPUOffload, StateDictType
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
except ImportError:  # pragma: no cover - FSDP optional
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    FSDP = None  # type: ignore[misc, assignment]  # optional torch import fallback
    ShardingStrategy = None  # type: ignore[misc, assignment]  # optional torch import fallback
    MixedPrecision = None  # type: ignore[misc, assignment]  # optional torch import fallback
    CPUOffload = None  # type: ignore[misc, assignment]  # optional torch import fallback
    BackwardPrefetch = None  # type: ignore[misc, assignment]  # optional torch import fallback
    StateDictType = None  # type: ignore[misc, assignment]  # optional torch import fallback
    ShardedGradScaler = None  # type: ignore[misc, assignment]  # optional torch import fallback
    dcp_save = None  # type: ignore[assignment]
    dcp_load = None  # type: ignore[assignment]
    FilePlanarPlacement = None  # type: ignore[assignment]
    DefaultSavePlanner = None  # type: ignore[misc, assignment]  # optional torch import fallback
    DefaultLoadPlanner = None  # type: ignore[misc, assignment]  # optional torch import fallback

logger = logging.getLogger("cadgenesis.training.fsdp")


@dataclass(frozen=True)
class FSDPConfig:
    """FSDP wrapping options."""

    sharding_strategy: str = "full_shard"
    mixed_precision: str = "bf16"
    cpu_offload: bool = False
    backward_prefetch: str = "backward_pre"

    @property
    def sharding_strategy_enum(self) -> Any:
        if ShardingStrategy is None:
            return None
        mapping = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        return mapping.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)

    @property
    def mixed_precision_policy(self) -> Any:
        if MixedPrecision is None:
            return None
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "none": None,
        }
        dtype = dtype_map.get(self.mixed_precision, torch.bfloat16)
        if dtype is None:
            return None
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    @property
    def cpu_offload_policy(self) -> Any:
        if CPUOffload is None or not self.cpu_offload:
            return None
        return CPUOffload(offload_params=True)

    @property
    def backward_prefetch_policy(self) -> Any:
        if BackwardPrefetch is None:
            return None
        return (
            BackwardPrefetch.BACKWARD_PRE
            if self.backward_prefetch == "backward_pre"
            else BackwardPrefetch.BACKWARD_POST
        )


def wrap_fsdp(
    model: Any,
    config: FSDPConfig | None = None,
    wrap_children: bool = True,
) -> Any:
    """Wrap ``model`` in FSDP; returns it unchanged outside DDP runs."""
    if FSDP is None or dist is None or not dist.is_available() or not dist.is_initialized():
        return model
    if dist.get_world_size() <= 1:
        return model
    if isinstance(model, FSDP):
        return model
    cfg = config or FSDPConfig()
    # device_id must match the process-group backend: with nccl it is the
    # local CUDA device; with gloo (CPU ranks) it must stay None.  (We reach
    # here only when a group is initialised, so the check is well-defined.)
    device_id = torch.cuda.current_device() if dist.get_backend() == "nccl" else None
    return FSDP(
        model,
        sharding_strategy=cfg.sharding_strategy_enum,
        mixed_precision=cfg.mixed_precision_policy,
        cpu_offload=cfg.cpu_offload_policy,
        backward_prefetch=cfg.backward_prefetch_policy,
        device_id=device_id,
    )


def fsdp_sharded_ckpt(
    model: Any,
    optimizer: Any,
    state_path: str,
    save: bool,
    rank: int = 0,
) -> None:
    """Save/load sharded FSDP state (optimizer + model) with torch.distributed.checkpoint."""
    if dcp_save is None or not isinstance(model, FSDP):
        raise RuntimeError("fsdp_sharded_ckpt requires FSDP + torch.distributed.checkpoint")
    if save:
        state = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }
        dcp_save(state, storage_writer=FilePlanarPlacement(state_path))
    else:
        state = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
        }
        dcp_load(state, storage_reader=FilePlanarPlacement(state_path))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])


def sharded_grad_scaler(enabled: bool = True) -> Any:
    """Gradient scaler that syncs across FSDP ranks (fp16 training)."""
    if ShardedGradScaler is None:
        return None
    return ShardedGradScaler(enabled=enabled)


def is_fsdp(model: Any) -> bool:
    """True when ``model`` is FSDP-wrapped (import-safe on CPU-only boxes)."""
    return FSDP is not None and isinstance(model, FSDP)


def fsdp_full_state_dict(model: Any) -> None:
    """Set FULL_STATE_DICT on ``model`` (collective; call on *all* ranks).

    After this, ``model.state_dict()`` / ``load_state_dict(...)`` operate on
    the unsharded full state on every rank.  No-op for non-FSDP models.
    """
    if not is_fsdp(model):
        return
    from torch.distributed.fsdp.api import StateDictType

    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT)


__all__ = [
    "FSDPConfig",
    "fsdp_full_state_dict",
    "fsdp_sharded_ckpt",
    "is_fsdp",
    "sharded_grad_scaler",
    "wrap_fsdp",
]
