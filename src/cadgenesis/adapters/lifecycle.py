"""cadgenesis.adapters.lifecycle
=============================
Adapter lifecycle management.

Tracks the state machine of every adapter (registered -> training -> candidate
-> promoted -> rolled_back / retired) and keeps it consistent with the
``status`` strings used by :mod:`cadgenesis.adapters.manager`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from cadgenesis.adapters.manager import AdapterMetadata


class AdapterLifecycleState(Enum):
    """Lifecycle states of an adapter."""

    REGISTERED = "registered"
    TRAINING = "training"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[AdapterLifecycleState, frozenset[AdapterLifecycleState]] = {
    AdapterLifecycleState.REGISTERED: frozenset(
        {
            AdapterLifecycleState.TRAINING,
            AdapterLifecycleState.CANDIDATE,
            AdapterLifecycleState.FAILED,
            AdapterLifecycleState.RETIRED,
        }
    ),
    AdapterLifecycleState.TRAINING: frozenset(
        {
            AdapterLifecycleState.CANDIDATE,
            AdapterLifecycleState.FAILED,
            AdapterLifecycleState.RETIRED,
        }
    ),
    AdapterLifecycleState.CANDIDATE: frozenset(
        {
            AdapterLifecycleState.TRAINING,
            AdapterLifecycleState.PROMOTED,
            AdapterLifecycleState.ROLLED_BACK,
            AdapterLifecycleState.FAILED,
            AdapterLifecycleState.RETIRED,
        }
    ),
    AdapterLifecycleState.PROMOTED: frozenset(
        {AdapterLifecycleState.ROLLED_BACK, AdapterLifecycleState.RETIRED}
    ),
    AdapterLifecycleState.ROLLED_BACK: frozenset(
        {
            AdapterLifecycleState.TRAINING,
            AdapterLifecycleState.CANDIDATE,
            AdapterLifecycleState.FAILED,
            AdapterLifecycleState.RETIRED,
        }
    ),
    AdapterLifecycleState.FAILED: frozenset(
        {AdapterLifecycleState.TRAINING, AdapterLifecycleState.RETIRED}
    ),
    AdapterLifecycleState.RETIRED: frozenset(),
}

_MANAGER_STATUS_TO_STATE: dict[str, AdapterLifecycleState] = {
    "candidate": AdapterLifecycleState.CANDIDATE,
    "promoted": AdapterLifecycleState.PROMOTED,
    "retired": AdapterLifecycleState.RETIRED,
    "rolled_back": AdapterLifecycleState.ROLLED_BACK,
}


@dataclass(frozen=True)
class LifecycleEvent:
    """A single recorded state transition."""

    adapter_id: str
    from_state: AdapterLifecycleState | None
    to_state: AdapterLifecycleState
    reason: str
    timestamp: float


class AdapterLifecycle:
    """State machine tracking adapter lifecycle transitions."""

    def __init__(self) -> None:
        self._states: dict[str, AdapterLifecycleState] = {}
        self._domains: dict[str, str] = {}
        self._events: list[LifecycleEvent] = []

    def register(self, adapter_id: str, domain: str) -> AdapterLifecycleState:
        """Register a new adapter in the REGISTERED state."""
        if adapter_id in self._states:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        self._states[adapter_id] = AdapterLifecycleState.REGISTERED
        self._domains[adapter_id] = domain
        self._events.append(
            LifecycleEvent(
                adapter_id=adapter_id,
                from_state=None,
                to_state=AdapterLifecycleState.REGISTERED,
                reason=f"registered for domain {domain!r}",
                timestamp=time.time(),
            )
        )
        return AdapterLifecycleState.REGISTERED

    def transition(self, adapter_id: str, to: AdapterLifecycleState, reason: str) -> None:
        """Move ``adapter_id`` to ``to``, enforcing the allowed-transition table."""
        current = self._states.get(adapter_id)
        if current is None:
            raise ValueError(f"adapter {adapter_id!r} is not registered")
        if to is current:
            raise ValueError(
                f"adapter {adapter_id!r} is already in state {to.value!r}; "
                "self-transitions are not allowed"
            )
        allowed = ALLOWED_TRANSITIONS[current]
        if to not in allowed:
            allowed_names = ", ".join(sorted(s.value for s in allowed)) or "none"
            raise ValueError(
                f"invalid transition {current.value!r} -> {to.value!r} for "
                f"adapter {adapter_id!r}; allowed targets: {allowed_names}"
            )
        self._states[adapter_id] = to
        self._events.append(
            LifecycleEvent(
                adapter_id=adapter_id,
                from_state=current,
                to_state=to,
                reason=reason,
                timestamp=time.time(),
            )
        )

    def state(self, adapter_id: str) -> AdapterLifecycleState:
        """Current lifecycle state of ``adapter_id`` (raises KeyError if unknown)."""
        return self._states[adapter_id]

    def domain(self, adapter_id: str) -> str | None:
        """Domain the adapter was registered under."""
        return self._domains.get(adapter_id)

    def history(self, adapter_id: str | None = None) -> list[LifecycleEvent]:
        """All recorded events, optionally filtered to one adapter."""
        if adapter_id is None:
            return list(self._events)
        return [event for event in self._events if event.adapter_id == adapter_id]

    def to_state(self, adapter: AdapterMetadata) -> AdapterLifecycleState:
        """Map a manager ``AdapterMetadata.status`` string to a lifecycle state."""
        return _MANAGER_STATUS_TO_STATE.get(adapter.status, AdapterLifecycleState.REGISTERED)
