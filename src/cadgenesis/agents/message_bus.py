"""cadgenesis.agents.message_bus
===============================
Agent communication bus — pub/sub messaging between role agents.

Messages are posted to named topics and delivered to every subscribed
handler.  The bus keeps a bounded history per topic so late subscribers and
audit trails can inspect past traffic.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

_Handler = Callable[["AgentMessage"], None]


@dataclass
class AgentMessage:
    """A single message travelling over the bus."""

    topic: str
    payload: dict[str, Any]
    sender: str = ""
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    priority: int = 0


class MessageBus:
    """Topic-based publish/subscribe bus with bounded history."""

    def __init__(self, history_limit: int = 512):
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        self.history_limit = history_limit
        self._subscribers: dict[str, list[_Handler]] = {}
        self._history: dict[str, list[AgentMessage]] = {}

    # ------------------------------------------------------------- subscribe

    def subscribe(self, topic: str, handler: _Handler) -> None:
        """Register ``handler`` to receive messages on ``topic``."""
        if not topic:
            raise ValueError("topic must be non-empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._subscribers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler: _Handler) -> bool:
        """Remove one handler; returns True when it was subscribed."""
        handlers = self._subscribers.get(topic)
        if not handlers or handler not in handlers:
            return False
        handlers.remove(handler)
        return True

    def subscriber_count(self, topic: str) -> int:
        return len(self._subscribers.get(topic, []))

    @property
    def topics(self) -> list[str]:
        return sorted(set(self._subscribers) | set(self._history))

    # -------------------------------------------------------------- publish

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        sender: str = "",
        priority: int = 0,
    ) -> AgentMessage:
        """Post a message, deliver it to subscribers, and archive it."""
        message = AgentMessage(
            topic=topic,
            payload=dict(payload),
            sender=sender,
            priority=priority,
        )
        for handler in list(self._subscribers.get(topic, [])):
            self._deliver(handler, message)
        history = self._history.setdefault(topic, [])
        history.append(message)
        if len(history) > self.history_limit:
            del history[: len(history) - self.history_limit]
        return message

    @staticmethod
    def _deliver(handler: _Handler, message: AgentMessage) -> None:
        """Call one subscriber; a faulty handler must not break the bus."""
        with suppress(Exception):
            handler(message)

    # -------------------------------------------------------------- history

    def history(self, topic: str) -> list[AgentMessage]:
        """Past messages on ``topic`` (oldest first)."""
        return list(self._history.get(topic, []))

    def all_history(self) -> dict[str, list[AgentMessage]]:
        return {topic: list(msgs) for topic, msgs in self._history.items()}

    def clear(self, topic: str | None = None) -> None:
        """Drop history (and optionally subscribers) for a topic."""
        if topic is None:
            self._history.clear()
            self._subscribers.clear()
        else:
            self._history.pop(topic, None)
            self._subscribers.pop(topic, None)

    def stats(self) -> dict[str, int]:
        return {topic: len(msgs) for topic, msgs in self._history.items()}
