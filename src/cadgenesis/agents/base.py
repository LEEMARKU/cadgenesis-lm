"""cadgenesis.agents.base
=======================
Shared foundation for the CADGenesis-LM multi-agent orchestration layer.

Defines :class:`AgentRequest` / :class:`AgentResult` (the message envelope
every agent speaks) and :class:`Agent` (the abstract base every role agent
implements).  This is the pure-Python orchestration counterpart to the torch
:class:`~cadgenesis.agents.multi_agent_system.MultiAgentSystem` that embeds
agent role heads inside the transformer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    """A task handed to an agent by the coordinator."""

    role: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("request requires a role")
        if not self.action:
            raise ValueError("request requires an action")


@dataclass
class AgentResult:
    """The outcome of an agent processing a request."""

    role: str
    action: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    task_id: str = ""

    @property
    def passed(self) -> bool:
        return self.ok


class Agent(ABC):
    """Base class for CAD role agents.

    Subclasses declare a ``role`` name and a set of ``actions`` they can
    perform, and implement :meth:`process`.
    """

    role: str = ""
    actions: tuple[str, ...] = ()

    def __init__(self) -> None:
        if not self.role:
            raise ValueError(f"{type(self).__name__} must define a role name")

    # ---------------------------------------------------------------- dispatch

    def can_handle(self, action: str) -> bool:
        """True when this agent implements ``action``."""
        return action in self.actions

    def handle(self, request: AgentRequest) -> AgentResult:
        """Validate the request then delegate to :meth:`process`."""
        if not self.can_handle(request.action):
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message=(f"agent {self.role!r} cannot handle action {request.action!r}"),
                task_id=request.task_id,
            )
        return self.process(request)

    @abstractmethod
    def process(self, request: AgentRequest) -> AgentResult:
        """Execute the request and return its result."""
        raise NotImplementedError

    # ------------------------------------------------------------------ misc

    def describe(self) -> dict[str, Any]:
        return {"role": self.role, "actions": list(self.actions)}
