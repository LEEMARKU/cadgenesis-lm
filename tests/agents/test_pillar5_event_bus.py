"""tests/agents/test_pillar5_event_bus.py
========================================
Unit tests for the Pillar 5 EventBus and SharedEventStore.
"""

from __future__ import annotations

from cadgenesis.agents.event_bus import Event, EventBus, SharedEventStore


def test_subscribe_and_publish_delivers():
    bus = EventBus()
    received = []
    bus.subscribe("design.created", received.append)
    event = bus.publish("design.created", {"id": 1})
    assert received == [event]
    assert received[0].payload == {"id": 1}


def test_priority_ordering():
    bus = EventBus(auto_flush=False)
    order = []
    bus.subscribe("t", lambda e: order.append(e.payload["value"]))
    bus.publish("t", {"value": "low"}, priority=1)
    bus.publish("t", {"value": "high"}, priority=10)
    bus.flush()
    assert order == ["high", "low"]


def test_wildcard_subscription():
    bus = EventBus()
    received = []
    bus.subscribe("*", received.append)
    bus.publish("anything", {"v": 1})
    assert len(received) == 1


def test_filter_fn():
    bus = EventBus()
    received = []
    bus.subscribe("t", received.append, filter_fn=lambda e: e.payload.get("ok"))
    bus.publish("t", {"ok": False})
    bus.publish("t", {"ok": True})
    assert len(received) == 1


def test_faulty_subscriber_isolated():
    bus = EventBus()
    received = []

    def boom(_event):
        raise RuntimeError("bad subscriber")

    bus.subscribe("t", boom)
    bus.subscribe("t", received.append)
    bus.publish("t", {})
    assert len(received) == 1


def test_request_response():
    bus = EventBus()
    bus.subscribe(
        "ask",
        lambda e: bus.respond(e, {"answer": 42}, sender="replier"),
        filter_fn=lambda e: e.kind == "request",
    )
    responses = bus.request("ask", {"q": "meaning"})
    assert len(responses) == 1
    assert responses[0].kind == "response"
    assert responses[0].payload == {"answer": 42}


def test_request_timeout_returns_empty():
    bus = EventBus()
    responses = bus.request("ask", {}, timeout=0.05)
    assert responses == []


def test_unsubscribe():
    bus = EventBus()
    received = []

    def handler(_event):
        received.append(1)

    bus.subscribe("t", handler)
    assert bus.unsubscribe("t", handler)
    bus.publish("t", {})
    assert received == []


def test_shared_event_store_query_and_replay():
    store = SharedEventStore()
    bus = EventBus(store=store)
    bus.publish("a", {"v": 1}, sender="s1")
    bus.publish("b", {"v": 2}, sender="s2")
    assert store.size == 2
    assert len(store.query(topic="a")) == 1
    assert len(store.query(sender="s1")) == 1
    assert len(store.query(kind="event")) == 2
    assert len(store.replay()) == 2


def test_event_store_restore_roundtrip():
    store = SharedEventStore()
    store.append(Event(topic="t", payload={"v": 1}))
    data = store.to_dict()
    restored = SharedEventStore()
    restored.restore(data)
    assert restored.size == 1
    assert restored.query(topic="t")[0].payload == {"v": 1}


def test_bus_stats_and_history():
    bus = EventBus()
    bus.subscribe("t", lambda e: None)
    bus.publish("t", {"v": 1})
    assert bus.stats()["subscribers"] >= 1
    assert bus.stats()["stored_events"] == 1
    assert len(bus.history("t")) == 1
