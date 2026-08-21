"""tests/training/test_distributed.py"""

from __future__ import annotations

import pytest

from cadgenesis.training.distributed import (
    DistributedContext,
    distribute_batch_size,
    get_context,
    is_main_process,
    launch,
    wrap_ddp,
)


def test_default_context_single_process():
    ctx = get_context()
    assert ctx.rank == 0
    assert ctx.world_size == 1
    assert ctx.is_main


def test_context_env_override(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    ctx = get_context()
    assert ctx.rank == 3
    assert ctx.local_rank == 1
    assert ctx.world_size == 8
    assert not ctx.is_main


def test_context_broadcast_single_process():
    ctx = DistributedContext()
    assert ctx.broadcast({"a": 1}) == {"a": 1}


def test_distribute_batch_size():
    assert distribute_batch_size(32, 1) == 32
    assert distribute_batch_size(32, 4) == 8
    assert distribute_batch_size(1, 4) == 1


def test_is_main_process_single():
    assert is_main_process() is True


def test_launch_single_process():
    result = launch(lambda ctx: ctx.rank)
    assert result == 0


def test_wrap_ddp_single_process_returns_model():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 4)
    assert wrap_ddp(model) is model
