"""tests/agents/test_message_bus.py
==================================
Unit tests for cadgenesis.agents.message_bus.
"""

from __future__ import annotations

from cadgenesis.agents.message_bus import AgentMessage, MessageBus


def test_publish_delivers_to_subscribers():
    bus = MessageBus()
    received: list[AgentMessage] = []
    bus.subscribe("topic.a", received.append)
    message = bus.publish("topic.a", {"k": 1}, sender="agent")
    assert received == [message]
    assert message.payload == {"k": 1}
    assert message.sender == "agent"


def test_subscribe_unsubscribe():
    bus = MessageBus()
    received = []
    bus.subscribe("topic.a", received.append)
    assert bus.unsubscribe("topic.a", received.append) is True
    assert bus.subscriber_count("topic.a") == 0


def test_history_bounded():
    bus = MessageBus(history_limit=3)
    for i in range(5):
        bus.publish("topic.a", {"i": i})
    history = bus.history("topic.a")
    assert len(history) == 3
    assert history[0].payload == {"i": 2}
    assert history[-1].payload == {"i": 4}


def test_faulty_subscriber_does_not_break_bus():
    bus = MessageBus()

    def boom(_msg):
        raise RuntimeError("boom")

    received = []
    bus.subscribe("topic.a", boom)
    bus.subscribe("topic.a", received.append)
    message = bus.publish("topic.a", {"k": 1})
    assert received == [message]


def test_topics_and_stats():
    bus = MessageBus()
    bus.publish("a", {})
    bus.publish("b", {})
    assert set(bus.topics) == {"a", "b"}
    assert bus.stats() == {"a": 1, "b": 1}


def test_clear():
    bus = MessageBus()
    bus.publish("a", {})
    bus.clear("a")
    assert bus.history("a") == []
    bus.publish("b", {})
    bus.clear()
    assert bus.history("b") == []
