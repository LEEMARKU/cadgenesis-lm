"""Tests for cadgenesis.adapters.lifecycle."""

import pytest

from cadgenesis.adapters.lifecycle import (
    ALLOWED_TRANSITIONS,
    AdapterLifecycle,
    AdapterLifecycleState,
)
from cadgenesis.adapters.manager import AdapterMetadata


def test_register_initial_state():
    lifecycle = AdapterLifecycle()
    state = lifecycle.register("aero_v1", "aerospace")
    assert state is AdapterLifecycleState.REGISTERED
    assert lifecycle.state("aero_v1") is AdapterLifecycleState.REGISTERED
    assert lifecycle.domain("aero_v1") == "aerospace"


def test_double_register_raises():
    lifecycle = AdapterLifecycle()
    lifecycle.register("aero_v1", "aerospace")
    with pytest.raises(ValueError):
        lifecycle.register("aero_v1", "aerospace")


def test_full_promotion_flow():
    lifecycle = AdapterLifecycle()
    lifecycle.register("aero_v1", "aerospace")
    lifecycle.transition("aero_v1", AdapterLifecycleState.TRAINING, "start training")
    lifecycle.transition("aero_v1", AdapterLifecycleState.CANDIDATE, "training done")
    lifecycle.transition("aero_v1", AdapterLifecycleState.PROMOTED, "passed eval")
    assert lifecycle.state("aero_v1") is AdapterLifecycleState.PROMOTED


def test_rolled_back_can_retrain():
    lifecycle = AdapterLifecycle()
    lifecycle.register("aero_v1", "aerospace")
    for to_state, reason in [
        (AdapterLifecycleState.TRAINING, "start"),
        (AdapterLifecycleState.CANDIDATE, "done"),
        (AdapterLifecycleState.PROMOTED, "pass"),
        (AdapterLifecycleState.ROLLED_BACK, "drift detected"),
        (AdapterLifecycleState.CANDIDATE, "retry"),
    ]:
        lifecycle.transition("aero_v1", to_state, reason)
    assert lifecycle.state("aero_v1") is AdapterLifecycleState.CANDIDATE


def test_invalid_transition_raises():
    lifecycle = AdapterLifecycle()
    lifecycle.register("aero_v1", "aerospace")
    with pytest.raises(ValueError, match="invalid transition"):
        lifecycle.transition("aero_v1", AdapterLifecycleState.PROMOTED, "too early")
    assert lifecycle.state("aero_v1") is AdapterLifecycleState.REGISTERED


def test_transition_to_same_state_raises():
    lifecycle = AdapterLifecycle()
    lifecycle.register("aero_v1", "aerospace")
    with pytest.raises(ValueError, match="already in state"):
        lifecycle.transition("aero_v1", AdapterLifecycleState.REGISTERED, "noop")


def test_transition_unregistered_raises():
    lifecycle = AdapterLifecycle()
    with pytest.raises(ValueError, match="not registered"):
        lifecycle.transition("ghost", AdapterLifecycleState.RETIRED, "gone")


def test_history_records_events():
    lifecycle = AdapterLifecycle()
    lifecycle.register("aero_v1", "aerospace")
    lifecycle.transition("aero_v1", AdapterLifecycleState.TRAINING, "start")
    events = lifecycle.history("aero_v1")
    assert len(events) == 2
    assert events[0].from_state is None
    assert events[0].to_state is AdapterLifecycleState.REGISTERED
    assert events[1].from_state is AdapterLifecycleState.REGISTERED
    assert events[1].to_state is AdapterLifecycleState.TRAINING
    assert events[1].reason == "start"
    assert events[1].timestamp >= events[0].timestamp


def test_history_filters_by_adapter():
    lifecycle = AdapterLifecycle()
    lifecycle.register("a", "aerospace")
    lifecycle.register("b", "automotive")
    assert len(lifecycle.history("a")) == 1
    assert len(lifecycle.history("b")) == 1
    assert len(lifecycle.history()) == 2


def test_allowed_transitions_cover_all_states():
    assert set(ALLOWED_TRANSITIONS) == set(AdapterLifecycleState)


def test_to_state_maps_manager_status():
    lifecycle = AdapterLifecycle()
    mapping = {
        "candidate": AdapterLifecycleState.CANDIDATE,
        "promoted": AdapterLifecycleState.PROMOTED,
        "retired": AdapterLifecycleState.RETIRED,
        "rolled_back": AdapterLifecycleState.ROLLED_BACK,
    }
    for status, expected in mapping.items():
        meta = AdapterMetadata(adapter_id="x", domain="y", status=status)
        assert lifecycle.to_state(meta) is expected
    unknown = AdapterMetadata(adapter_id="x", domain="y", status="weird")
    assert lifecycle.to_state(unknown) is AdapterLifecycleState.REGISTERED
