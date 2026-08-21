"""cadgenesis.agents.event_bus
============================
Pillar 5 communication layer built on top of the existing :class:`MessageBus`.

Provides:

* :class:`Event` — a typed, priority-ordered message envelope.
* :class:`EventBus` — publish/subscribe with **priority-ordered delivery**,
  wildcard topics, broadcast and request/response correlation.
* :class:`SharedEventStore` — durable in-memory event log with replay and
  query by topic / sender / time window.

The existing ``MessageBus`` (synchronous FIFO pub/sub) remains untouched for
backward compatibility; :class:`EventBus` composes it for advanced routing.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.agents.message_bus import MessageBus


@dataclass
class Event:
    """A priority-ordered message envelope."""

    topic: str
    payload: dict[str, Any]
    sender: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    correlation_id: str = ""
    kind: str = "event"  # "event" | "request" | "response"

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "sender": self.sender,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "kind": self.kind,
        }


@dataclass
class Subscription:
    """A registered event handler with an optional topic filter."""

    topic: str
    handler: Callable[[Event], None]
    filter: Callable[[Event], bool] | None = None


class EventBus:
    """Priority-ordered publish/subscribe bus with request/response support.

    Delivery order follows ``priority`` (higher first), then arrival order for
    equal priorities.  Wildcard ``"*"`` subscriptions receive every event.
    """

    def __init__(
        self,
        history_limit: int = 1024,
        store: SharedEventStore | None = None,
        base_bus: MessageBus | None = None,
        auto_flush: bool = True,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        self._base_bus = base_bus or MessageBus(history_limit=history_limit)
        self._subs: list[Subscription] = []
        self._pending_events: list[tuple[int, int, Event]] = []
        self._sequence = 0
        self._pending_requests: dict[str, list[Event]] = {}
        self._lock = threading.RLock()
        self.store = store or SharedEventStore()
        # ``auto_flush=False`` batches published events so that priority
        # ordering is observable across a batch (see :meth:`flush`).
        self._auto_flush = auto_flush
        # Internal capture: route every response to its pending request.
        self.subscribe(
            "*",
            self.capture_response,
            filter_fn=lambda event: event.kind == "response",
        )

    # ------------------------------------------------------------ subscription

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], None],
        filter_fn: Callable[[Event], bool] | None = None,
    ) -> None:
        if not topic:
            raise ValueError("subscription requires a topic")
        if not callable(handler):
            raise ValueError("handler must be callable")
        with self._lock:
            self._subs.append(Subscription(topic, handler, filter_fn))

    def unsubscribe(self, topic: str, handler: Callable[[Event], None]) -> bool:
        with self._lock:
            before = len(self._subs)
            self._subs = [s for s in self._subs if not (s.topic == topic and s.handler is handler)]
            return len(self._subs) < before

    def subscriber_count(self, topic: str) -> int:
        with self._lock:
            if topic == "*":
                return sum(1 for s in self._subs if s.topic == "*")
            return sum(1 for s in self._subs if s.topic == topic or s.topic == "*")

    # ----------------------------------------------------------------- publish

    def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        sender: str = "",
        priority: int = 0,
        correlation_id: str = "",
        kind: str = "event",
    ) -> Event:
        """Enqueue an event; delivery is processed immediately in priority order."""
        event = Event(
            topic=topic,
            payload=payload or {},
            sender=sender,
            priority=priority,
            correlation_id=correlation_id,
            kind=kind,
        )
        with self._lock:
            self._sequence += 1
            heapq.heappush(self._pending_events, (-event.priority, self._sequence, event))
        if self._auto_flush:
            self._drain()
        self.store.append(event)
        return event

    def flush(self) -> None:
        """Deliver all queued events in priority order.

        With ``auto_flush=False``, call :meth:`flush` after publishing a batch
        to observe cross-event priority ordering.
        """
        self._drain()

    def broadcast(
        self, topic: str, payload: dict[str, Any] | None = None, sender: str = ""
    ) -> Event:
        """Alias of :meth:`publish` with wildcard fan-out semantics."""
        return self.publish(topic, payload, sender=sender)

    def _drain(self) -> None:
        while True:
            with self._lock:
                if not self._pending_events:
                    return
                _, _, event = heapq.heappop(self._pending_events)
            self._deliver(event)

    def _deliver(self, event: Event) -> None:
        for sub in list(self._subs):
            if sub.topic != "*" and sub.topic != event.topic:
                continue
            if sub.filter is not None and not sub.filter(event):
                continue
            try:
                sub.handler(event)
            except Exception:
                continue

    # ------------------------------------------------------- request / response

    def request(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        sender: str = "",
        timeout: float | None = None,
        min_responses: int = 1,
    ) -> list[Event]:
        """Publish a request and collect responses with the same correlation id.

        Waits until ``min_responses`` replies arrive (or ``timeout`` elapses).
        """
        correlation_id = uuid.uuid4().hex[:16]
        self.publish(
            topic,
            payload,
            sender=sender,
            priority=10,
            correlation_id=correlation_id,
            kind="request",
        )
        deadline = None if timeout is None else time.time() + timeout
        responses: list[Event] = []
        while len(responses) < min_responses:
            if deadline is not None and time.time() > deadline:
                break
            self.flush()
            collected = self._take_responses(correlation_id)
            responses.extend(collected)
            if len(responses) >= min_responses:
                break
            time.sleep(0.001)
        responses.extend(self._take_responses(correlation_id))
        return responses

    def respond(self, request_event: Event, payload: dict[str, Any], sender: str = "") -> Event:
        """Reply to a previously published request."""
        return self.publish(
            request_event.topic,
            payload,
            sender=sender,
            priority=5,
            correlation_id=request_event.correlation_id,
            kind="response",
        )

    def _take_responses(self, correlation_id: str) -> list[Event]:
        with self._lock:
            buffered = self._pending_requests.get(correlation_id, [])
            if not buffered:
                return []
            del self._pending_requests[correlation_id]
            return buffered

    def capture_response(self, event: Event) -> None:
        """Store a response event keyed by correlation id (for async repliers)."""
        if not event.correlation_id:
            return
        with self._lock:
            self._pending_requests.setdefault(event.correlation_id, []).append(event)

    # ----------------------------------------------------------------- history

    def history(self, topic: str) -> list[Event]:
        """Recent events for a topic (oldest first) from the shared store."""
        return self.store.query(topic=topic)

    def stats(self) -> dict[str, int]:
        return {
            "subscribers": len(self._subs),
            "stored_events": self.store.size,
        }


class SharedEventStore:
    """Durable in-process event log with replay and time-window queries."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.RLock()

    def append(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)

    def query(
        self,
        topic: str | None = None,
        sender: str | None = None,
        since: float | None = None,
        until: float | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Query the event log by topic / sender / time window / kind."""
        with self._lock:
            results: list[Event] = []
            for event in self._events:
                if topic is not None and event.topic != topic:
                    continue
                if sender is not None and event.sender != sender:
                    continue
                if since is not None and event.timestamp < since:
                    continue
                if until is not None and event.timestamp > until:
                    continue
                if kind is not None and event.kind != kind:
                    continue
                results.append(event)
                if limit is not None and len(results) >= limit:
                    break
            return results

    def replay(self, topic: str | None = None) -> list[Event]:
        """Replay matching events in order (use to rebuild state)."""
        return self.query(topic=topic)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def to_dict(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._events]

    def restore(self, events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._events = [Event(**e) for e in events]
