"""Tests for cadgenesis.adapters.rollback."""

import pytest
import torch
import torch.nn as nn

from cadgenesis.adapters.rollback import AdapterRollback


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def test_snapshot_returns_checkpoint_id(tmp_path):
    rb = AdapterRollback()
    checkpoint_id = rb.snapshot(make_model(), "aero_v1", tmp_path, reason="pre-training")
    assert checkpoint_id.startswith("aero_v1_")
    assert checkpoint_id.endswith(".pt")
    assert (tmp_path / checkpoint_id).exists()


def test_snapshot_requires_base_dir():
    rb = AdapterRollback()
    try:
        rb.snapshot(make_model(), "aero_v1")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when base_dir is missing")


def test_rollback_round_trip(tmp_path):
    model = make_model()
    rb = AdapterRollback()
    checkpoint_id = rb.snapshot(model, "aero_v1", tmp_path, reason="pre-training")
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model[0].weight.add_(2.0)
    path = rb.rollback("aero_v1", checkpoint_id, tmp_path, model)
    assert path == str(tmp_path / checkpoint_id)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_rollback_latest_when_checkpoint_omitted(tmp_path):
    model = make_model()
    rb = AdapterRollback()
    rb.snapshot(model, "aero_v1", tmp_path, reason="first")
    with torch.no_grad():
        model[0].weight.mul_(2.0)
    rb.snapshot(model, "aero_v1", tmp_path, reason="second")
    with torch.no_grad():
        model[0].weight.add_(5.0)
    path = rb.rollback("aero_v1", None, tmp_path, model)
    expected = torch.load(path, weights_only=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected["state_dict"][key])


def test_rollback_without_model_returns_path(tmp_path):
    model = make_model()
    rb = AdapterRollback()
    checkpoint_id = rb.snapshot(model, "aero_v1", tmp_path)
    path = rb.rollback("aero_v1", checkpoint_id, tmp_path)
    assert path == str(tmp_path / checkpoint_id)


def test_rollback_unknown_checkpoint_raises(tmp_path):
    rb = AdapterRollback()
    with pytest.raises(FileNotFoundError):
        rb.rollback("aero_v1", "aero_v1_missing.pt", tmp_path)


def test_rollback_no_checkpoints_raises(tmp_path):
    rb = AdapterRollback()
    with pytest.raises(FileNotFoundError):
        rb.rollback("aero_v1", None, tmp_path)


def test_list_checkpoints_ordered(tmp_path):
    rb = AdapterRollback()
    first = rb.snapshot(make_model(), "aero_v1", tmp_path)
    second = rb.snapshot(make_model(), "aero_v1", tmp_path)
    checkpoints = rb.list_checkpoints("aero_v1", tmp_path)
    assert len(checkpoints) == 2
    assert checkpoints[0].endswith(first)
    assert checkpoints[1].endswith(second)


def test_list_checkpoints_ignores_other_adapters(tmp_path):
    rb = AdapterRollback()
    rb.snapshot(make_model(), "aero_v1", tmp_path)
    rb.snapshot(make_model(), "auto_v2", tmp_path)
    assert len(rb.list_checkpoints("aero_v1", tmp_path)) == 1
    assert len(rb.list_checkpoints("auto_v2", tmp_path)) == 1
