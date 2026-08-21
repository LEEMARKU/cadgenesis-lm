"""cadgenesis.reasoning.symbolic_planner
========================================
Symbolic (STRIPS-lite) planning over engineering intent (v6.0, Pillar 7).

A state is a plain ``dict`` of boolean/valued facts; an
:class:`PlanningOperator` declares a precondition predicate, an effect
transformer and a cost.  :class:`SymbolicPlanner` performs forward
state-space search from an initial state to a goal, returns the operator
chain with per-state snapshots, and can:

* **plan** — best-first (cost-ordered BFS) search with cycle pruning;
* **decompose** — goal regression into sub-goals via operator preconditions;
* **to_cad_plan** — map the operator chain onto the existing
  :class:`~cadgenesis.reasoning.planner.CADPlan` (dependency + execution
  ordering preserved), bridging into the workflow planner contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.reasoning.planner import CADPlan, PlanningStep

_ACTION_MAP = {
    "sketch": "sketch",
    "model": "model",
    "constrain": "constrain",
    "assemble": "assemble",
    "simulate": "simulate",
    "validate": "validate",
    "manufacture": "manufacture",
    "export": "export",
}


@dataclass
class PlanningOperator:
    """A symbolic planning operator with preconditions and effects."""

    name: str
    precondition: Callable[[dict[str, Any]], bool]
    effect: Callable[[dict[str, Any]], dict[str, Any]]
    cost: float = 1.0
    description: str = ""
    action: str = "model"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operator name must be non-empty")
        if not callable(self.precondition):
            raise TypeError("precondition must be callable")
        if not callable(self.effect):
            raise TypeError("effect must be callable")
        if self.cost < 0:
            raise ValueError("cost must be >= 0")
        if self.action not in _ACTION_MAP:
            raise ValueError(f"invalid action {self.action!r}")

    def applicable(self, state: dict[str, Any]) -> bool:
        return bool(self.precondition(state))

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        new_state = dict(state)
        new_state.update(self.effect(state))
        return new_state


@dataclass
class SymbolicPlan:
    """Result of symbolic planning."""

    goal: str
    operators: list[str] = field(default_factory=list)
    states: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0
    explored: int = 0

    @property
    def solved(self) -> bool:
        return bool(self.operators)

    def dependency_graph(self) -> list[tuple[int, int]]:
        """Index pairs ``(i, j)`` where operator ``j`` re-uses a fact set by ``i``.

        The effect of ``operators[i]`` sets at least one fact that
        ``operators[j]``'s precondition requires.
        """
        edges: list[tuple[int, int]] = []
        produced: dict[str, int] = {}
        for index, _ in enumerate(self.operators):
            for fact in self.states[index]:
                if fact in produced and produced[fact] < index:
                    edges.append((produced[fact], index))
                produced[fact] = index
        return edges

    def execution_order(self) -> list[str]:
        """The operator chain in (already sequential) execution order."""
        return list(self.operators)

    def to_cad_plan(self, goal: str) -> CADPlan:
        """Convert the operator chain into a dependency-ordered CADPlan."""
        plan = CADPlan(goal=goal)
        facts_produced: dict[str, str] = {}
        for index, name in enumerate(self.operators):
            step_id = f"step{index + 1}"
            depends_on: list[str] = []
            facts_needed = [fact for fact in self.states[index] if fact not in facts_produced]
            for fact in facts_needed:
                producer = facts_produced.get(fact)
                if producer is not None and producer != step_id:
                    depends_on.append(producer)
            action = _ACTION_MAP.get(name.split(":")[0], "model")
            plan.add_step(
                PlanningStep(
                    id=step_id,
                    action=action,
                    description=self._operator_description(name),
                    depends_on=depends_on,
                    params={"operator": name},
                )
            )
            for fact in self.states[index]:
                facts_produced.setdefault(fact, step_id)
        return plan

    def _operator_description(self, name: str) -> str:
        return f"symbolic step {name}"

    def summary(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "solved": self.solved,
            "steps": len(self.operators),
            "cost": self.cost,
            "explored": self.explored,
            "operators": list(self.operators),
        }


class SymbolicPlanner:
    """Forward state-space symbolic planner over engineering intent."""

    def __init__(
        self,
        operators: list[PlanningOperator] | None = None,
        max_depth: int = 12,
        max_states: int = 10_000,
    ) -> None:
        self._operators: list[PlanningOperator] = []
        if operators:
            for operator in operators:
                self.register(operator)
        self.max_depth = max_depth
        self.max_states = max_states

    def register(self, operator: PlanningOperator) -> None:
        if any(o.name == operator.name for o in self._operators):
            raise ValueError(f"operator {operator.name!r} already registered")
        self._operators.append(operator)

    def get(self, name: str) -> PlanningOperator | None:
        return next((o for o in self._operators if o.name == name), None)

    @property
    def operator_names(self) -> list[str]:
        return [o.name for o in self._operators]

    def __len__(self) -> int:
        return len(self._operators)

    # ------------------------------------------------------------- planning

    def plan(
        self,
        goal: str,
        initial: dict[str, Any],
        goal_test: Callable[[dict[str, Any]], bool],
        max_depth: int | None = None,
        max_states: int | None = None,
    ) -> SymbolicPlan:
        """Best-first (cost-ordered) forward search for ``goal_test``.

        States are hashed by their sorted facts; revisited states are pruned.
        Returns an empty :class:`SymbolicPlan` when no plan is found within
        the depth/state budget.
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("goal must be a non-empty string")
        depth_limit = self.max_depth if max_depth is None else max_depth
        state_limit = self.max_states if max_states is None else max_states
        if depth_limit < 1:
            raise ValueError("max_depth must be >= 1")

        from heapq import heappop, heappush

        start = dict(initial)
        explored = 0
        queue: list[tuple[float, int, dict[str, Any], list[str]]] = []
        visited: set[tuple[tuple[str, Any], ...]] = {tuple(sorted(start.items()))}
        heappush(queue, (0.0, 0, start, []))
        while queue and explored < state_limit:
            cost, _, state, chain = heappop(queue)
            explored += 1
            if goal_test(state):
                return SymbolicPlan(
                    goal=goal,
                    operators=chain,
                    states=self._chain_states(start, chain),
                    cost=cost,
                    explored=explored,
                )
            if len(chain) >= depth_limit:
                continue
            for operator in sorted(self._operators, key=lambda o: (o.cost, o.name)):
                if not operator.applicable(state):
                    continue
                next_state = operator.apply(state)
                key = tuple(sorted(next_state.items()))
                if key in visited:
                    continue
                visited.add(key)
                heappush(
                    queue,
                    (
                        cost + operator.cost,
                        explored,
                        next_state,
                        [*chain, operator.name],
                    ),
                )
        return SymbolicPlan(
            goal=goal,
            operators=[],
            states=[],
            cost=0.0,
            explored=explored,
        )

    def _chain_states(
        self,
        initial: dict[str, Any],
        chain: list[str],
    ) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = [dict(initial)]
        current = dict(initial)
        for name in chain:
            operator = self.get(name)
            if operator is not None:
                current = operator.apply(current)
            states.append(dict(current))
        return states

    # ---------------------------------------------------------- decomposition

    def decompose(
        self,
        goal: str,
        goal_test: Callable[[dict[str, Any]], bool],
        initial: dict[str, Any],
        max_depth: int = 6,
    ) -> list[str]:
        """Goal regression: the operator chain that *could* reach the goal.

        Searches from the initial state using only preconditions (effects are
        applied, but no full solve is required); returns the operator names
        whose combined application makes ``goal_test`` satisfiable, or ``[]``.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        result = self.plan(goal, initial, goal_test, max_depth=max_depth)
        return result.operators

    # ---------------------------------------------------------------- misc

    def summary(self) -> dict[str, Any]:
        return {
            "operators": self.operator_names,
            "max_depth": self.max_depth,
            "max_states": self.max_states,
        }


__all__ = [
    "PlanningOperator",
    "SymbolicPlan",
    "SymbolicPlanner",
]
