"""tests/continual_learning/test_updater.py
=========================================
Unit tests for the incremental checkpoint updater.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cadgenesis.continual_learning.updater import ModelUpdater


def _make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_save_and_load_round_trip(tmp_path: Path):
    model = _make_model()
    updater = ModelUpdater()
    path = updater.save_incremental(model, "task_1", tmp_path, metadata={"step": 3})
    assert path.name == "task_task_1_1.pt"
    assert path.is_file()
    state, meta = updater.load_latest(tmp_path)
    assert meta["task_id"] == "task_1"
    assert meta["version"] == 1
    assert meta["step"] == 3
    assert "timestamp" in meta
    for key, value in model.state_dict().items():
        assert torch.equal(state[key], value)


def test_version_bumps_per_save(tmp_path: Path):
    model = _make_model()
    updater = ModelUpdater()
    updater.save_incremental(model, "task_1", tmp_path)
    updater.save_incremental(model, "task_1", tmp_path, metadata={"step": 5})
    versions = sorted(c["version"] for c in updater.list_checkpoints(tmp_path))
    assert versions == [1, 2]
    _state, meta = updater.load_latest(tmp_path)
    assert meta["version"] == 2
    assert meta["step"] == 5


def test_multiple_tasks_and_load_latest_for(tmp_path: Path):
    model_a = _make_model()
    model_b = _make_model()
    updater = ModelUpdater()
    updater.save_incremental(model_a, "task_a", tmp_path)
    updater.save_incremental(model_b, "task_b", tmp_path)
    updater.save_incremental(model_a, "task_a", tmp_path)
    _state, meta = updater.load_latest(tmp_path)
    assert meta["task_id"] == "task_a"
    assert meta["version"] == 2
    state_a, meta_a = updater.load_latest_for(tmp_path, "task_a")
    assert meta_a["version"] == 2
    assert torch.equal(state_a["0.weight"], model_a.state_dict()["0.weight"])
    assert len(updater.list_checkpoints(tmp_path)) == 3


def test_load_latest_empty_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ModelUpdater().load_latest(tmp_path)


def test_weights_only_round_trip(tmp_path: Path):
    model = _make_model()
    updater = ModelUpdater()
    updater.save_incremental(model, "t1", tmp_path)
    path = updater.checkpoint_path(tmp_path, "t1", 1)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["2.bias"], model.state_dict()["2.bias"])


def test_checkpoint_path_sanitizes_task_id(tmp_path: Path):
    path = ModelUpdater().checkpoint_path(tmp_path, "task/1:buggy", 1)
    assert path.name == "task_task_1_buggy_1.pt"


def test_list_checkpoints_empty(tmp_path: Path):
    assert ModelUpdater().list_checkpoints(tmp_path) == []
