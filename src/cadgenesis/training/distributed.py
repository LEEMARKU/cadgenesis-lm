"""
cadgenesis.training.distributed
===============================
Distributed data-parallel (DDP) launcher and helpers for CADGenesis-LM.

``launch`` initializes the process group (NCCL/GLOO), wraps the model in
``DistributedDataParallel``, and runs a per-rank worker callable with a
``DistributedContext`` (rank, world_size, local_rank, is_main).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

try:
    import torch
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DistributedContext:
    """Per-rank training context."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    is_main: bool = True

    def broadcast(self, obj: Any) -> Any:
        """Broadcast ``obj`` from rank 0 when a process group is active."""
        if dist is None or not dist.is_available() or self.world_size <= 1:
            return obj
        output = [obj]
        dist.broadcast_object_list(output, src=0)
        return output[0]


def _env(name: str, default: Any = 0) -> int:
    return int(os.environ.get(name, default))


def get_context() -> DistributedContext:
    """Build a context from environment variables (torchrun sets these)."""
    world_size = _env("WORLD_SIZE", 1)
    rank = _env("RANK", 0)
    local_rank = _env("LOCAL_RANK", 0)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=os.environ.get("BACKEND", "nccl"),
        is_main=rank == 0,
    )


def init_process_group(
    backend: str | None = None,
    init_method: str | None = None,
    timeout_minutes: float = 30.0,
) -> DistributedContext:
    """Initialize the default process group and return the rank context.

    Falls back to single-process (no-op) when torch.distributed is
    unavailable or already initialized.
    """
    if dist is None or not dist.is_available():
        return DistributedContext()
    if dist.is_initialized():
        ctx = get_context()
        return ctx
    # Not launched via torchrun/mpirun (no RANK env): run single-process.
    if "RANK" not in os.environ and init_method is None:
        return DistributedContext()
    backend = backend or os.environ.get("BACKEND", "nccl")
    init_method = init_method or os.environ.get("INIT_METHOD", "env://")
    dist.init_process_group(backend=backend, init_method=init_method)
    return get_context()


def is_main_process() -> bool:
    """True on rank 0 or in single-process runs."""
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def wrap_ddp(model: Any, ctx: DistributedContext | None = None) -> Any:
    """Wrap ``model`` in DistributedDataParallel when world_size > 1."""
    if torch is None or dist is None or not dist.is_available():
        return model
    if ctx is None:
        ctx = get_context()
    if ctx.world_size <= 1:
        return model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ctx.local_rank] if torch.cuda.is_available() else None,
        output_device=ctx.local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=True,
    )


def destroy_process_group() -> None:
    """Tear down the process group if initialized."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def launch(
    worker: Callable[[DistributedContext], Any],
    backend: str | None = None,
    init_method: str | None = None,
    cleanup: bool = True,
) -> Any:
    """Run ``worker(ctx)`` on each rank with a ready process group.

    Intended to be invoked via ``torchrun``/``mpirun``; degrades to a
    single-process call when run standalone.
    """
    ctx = init_process_group(backend=backend, init_method=init_method)
    try:
        return worker(ctx)
    finally:
        if cleanup:
            destroy_process_group()


def distribute_batch_size(batch_size: int, world_size: int) -> int:
    """Per-rank batch size for a global target with ``world_size`` workers."""
    if world_size <= 1:
        return batch_size
    return max(1, (batch_size + world_size - 1) // world_size)


__all__ = [
    "DistributedContext",
    "destroy_process_group",
    "distribute_batch_size",
    "get_context",
    "init_process_group",
    "is_main_process",
    "launch",
    "wrap_ddp",
]
