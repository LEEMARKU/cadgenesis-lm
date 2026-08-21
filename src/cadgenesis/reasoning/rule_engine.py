"""cadgenesis.reasoning.rule_engine
==================================
Rule-based reasoning engine for CADGenesis-LM v6.0.

Rules are lightweight condition/action pairs evaluated against a fact
``context`` (a plain ``dict``).  The engine supports priority ordering,
forward-chained execution (rules may mutate the context and fire follow-up
rules) and structured reporting of triggered rules and violations.

Design heuristics (wall thickness, draft angles, manufacturability limits,
standard compliance) are naturally expressed as rules, so the engine is the
shared substrate for :mod:`cadgenesis.reasoning.manufacturing_rules` and the
:mod:`cadgenesis.reasoning.validator` orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Severity levels, ordered from mild to severe.
SEVERITY_ORDER: tuple[str, ...] = ("info", "warning", "error", "critical")

VALID_SEVERITIES = set(SEVERITY_ORDER)


@dataclass
class Rule:
    """A single condition/action production rule."""

    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: Callable[[dict[str, Any]], str | None] | None = None
    description: str = ""
    severity: str = "info"
    priority: int = 0
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Rule name must be non-empty")
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"invalid severity {self.severity!r}; expected one of {sorted(VALID_SEVERITIES)}"
            )
        if not callable(self.condition):
            raise TypeError("Rule condition must be callable")
        if self.action is not None and not callable(self.action):
            raise TypeError("Rule action must be callable or None")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Rule version must be a non-empty string")

    def severity_index(self) -> int:
        return SEVERITY_ORDER.index(self.severity)

    def concludes(self) -> str | None:
        """The fact this rule can establish (from ``meta["concludes"]``)."""
        return self.meta.get("concludes")

    def requires(self) -> list[str]:
        """Sub-goals needed before this rule applies (``meta["requires"]``)."""
        return list(self.meta.get("requires") or [])


@dataclass
class RuleResult:
    """Outcome of evaluating a rule against a context."""

    rule: Rule
    triggered: bool
    message: str | None = None
    context_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rule.name

    @property
    def severity(self) -> str:
        return self.rule.severity


@dataclass
class Proof:
    """Goal-directed (backward-chaining) proof of a fact."""

    goal: str
    established: bool
    depth: int
    steps: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "established": self.established,
            "depth": self.depth,
            "steps": list(self.steps),
            "trace": list(self.trace),
        }


def make_rule(
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
    description: str = "",
    severity: str = "info",
    priority: int = 0,
    action: Callable[[dict[str, Any]], str | None] | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    version: str = "1.0.0",
) -> Rule:
    """Convenience factory for a predicate-only rule (no action)."""
    return Rule(
        name=name,
        condition=predicate,
        action=action,
        description=description,
        severity=severity,
        priority=priority,
        tags=tags or [],
        meta=meta or {},
        version=version,
    )


class RuleEngine:
    """Evaluates a set of rules against fact contexts.

    Supports forward chaining: when a rule's ``action`` mutates the context,
    ``run()`` re-evaluates rules that have not yet fired this run, allowing
    cascading conclusions.  Each rule fires at most once per ``run()`` call.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = []
        self._by_name: dict[str, Rule] = {}
        if rules:
            self.add_rules(rules)

    # ------------------------------------------------------------------ rules

    def add_rule(self, rule: Rule) -> None:
        if rule.name in self._by_name:
            raise ValueError(f"A rule named {rule.name!r} is already registered")
        self._rules.append(rule)
        self._by_name[rule.name] = rule

    def add_rules(self, rules: list[Rule]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def remove_rule(self, name: str) -> bool:
        if name not in self._by_name:
            return False
        self._rules = [r for r in self._rules if r.name != name]
        del self._by_name[name]
        return True

    def get_rule(self, name: str) -> Rule | None:
        return self._by_name.get(name)

    def rule_names(self) -> list[str]:
        return [r.name for r in self._rules]

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    # --------------------------------------------------------------- evaluate

    def evaluate(
        self,
        context: dict[str, Any],
        rule_names: list[str] | None = None,
    ) -> list[RuleResult]:
        """Evaluate the (selected) rules once against ``context``.

        Returns one ``RuleResult`` per rule in priority order.  ``context`` is
        mutated by rule actions only during :meth:`run`; a single evaluation
        applies the rule's condition without its action side effects.
        """
        rules = self._select(rule_names)
        rules = sorted(rules, key=lambda r: (-r.priority, r.name))
        results: list[RuleResult] = []
        for rule in rules:
            triggered = bool(rule.condition(context))
            message = None
            if triggered and rule.action is not None:
                message = rule.action(context)
            results.append(RuleResult(rule, triggered, message, dict(context)))
        return results

    def evaluate_single(self, name: str, context: dict[str, Any]) -> RuleResult | None:
        rule = self.get_rule(name)
        if rule is None:
            return None
        triggered = bool(rule.condition(context))
        message = rule.action(context) if triggered and rule.action else None
        return RuleResult(rule, triggered, message, dict(context))

    def run(
        self,
        context: dict[str, Any],
        max_rounds: int = 10,
    ) -> list[RuleResult]:
        """Forward-chain: repeatedly evaluate until a fixed point.

        Actions may mutate ``context``.  A rule fires at most once per run, so
        evaluation terminates after at most ``len(rules)`` rounds.  Returns all
        results across rounds, ordered by round.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        fired: set[str] = set()
        results: list[RuleResult] = []
        for _ in range(min(max_rounds, len(self._rules) or 1)):
            progressed = False
            for rule in sorted(self._rules, key=lambda r: (-r.priority, r.name)):
                if rule.name in fired:
                    continue
                if rule.condition(context):
                    fired.add(rule.name)
                    progressed = True
                    message = rule.action(context) if rule.action else None
                    results.append(RuleResult(rule, True, message, dict(context)))
            if not progressed:
                break
        return results

    def violations(
        self, context: dict[str, Any], min_severity: str = "warning"
    ) -> list[RuleResult]:
        """Return triggered results whose severity is at least ``min_severity``."""
        min_idx = SEVERITY_ORDER.index(min_severity)
        return [
            r for r in self.evaluate(context) if r.triggered and r.rule.severity_index() >= min_idx
        ]

    def summary(self, context: dict[str, Any]) -> dict[str, Any]:
        """Aggregate counts of triggered rules by severity."""
        results = self.evaluate(context)
        counts = {sev: 0 for sev in VALID_SEVERITIES}
        for r in results:
            if r.triggered:
                counts[r.rule.severity] += 1
        return {
            "total_rules": len(results),
            "triggered": sum(1 for r in results if r.triggered),
            "by_severity": counts,
            "fired": [r.name for r in results if r.triggered],
        }

    # --------------------------------------------------------- versioning

    def snapshot(self) -> dict[str, Any]:
        """Versioned snapshot of the rule set (names + versions)."""
        return {
            "rules": sorted(
                (
                    {
                        "name": rule.name,
                        "version": rule.version,
                        "severity": rule.severity,
                        "priority": rule.priority,
                    }
                    for rule in self._rules
                ),
                key=lambda r: r["name"],
            ),
            "total": self.rule_count,
        }

    def by_version(self, version: str) -> list[Rule]:
        """Rules registered with a specific version string."""
        return [r for r in self._rules if r.version == version]

    def diff(self, other: RuleEngine) -> dict[str, Any]:
        """Compare two rule sets: added / removed / version-changed rules."""
        left = {r.name: r for r in self._rules}
        right = {r.name: r for r in other._rules}
        added = sorted(set(right) - set(left))
        removed = sorted(set(left) - set(right))
        changed = sorted(
            name for name in set(left) & set(right) if left[name].version != right[name].version
        )
        return {"added": added, "removed": removed, "changed": changed}

    # ------------------------------------------------------ backward chaining

    def prove(
        self,
        goal: str,
        context: dict[str, Any],
        depth_limit: int = 8,
    ) -> Proof:
        """Backward-chain to establish ``goal`` against ``context``.

        A goal is established when the context already satisfies it, or when a
        rule whose ``meta["concludes"] == goal`` fires and every fact in its
        ``meta["requires"]`` list is itself provable (recursively).  Conditions
        are evaluated against the context *plus* facts already proven during
        this search, so a rule may check for the presence of its own
        sub-conclusions.  The result is a :class:`Proof` carrying the rule
        chain and a human-readable trace.  The context is never mutated by
        this method.
        """
        if depth_limit < 1:
            raise ValueError("depth_limit must be >= 1")
        trace: list[str] = []
        steps: list[str] = []
        derived: set[str] = set()

        def _context_establishes(fact: str) -> bool:
            if fact in derived:
                return True
            value = context.get(fact)
            return bool(value) if not isinstance(value, bool) else value

        def _merged() -> dict[str, Any]:
            merged = dict(context)
            for fact in derived:
                merged[fact] = True
            return merged

        def _search(fact: str, depth: int, chain: list[str]) -> bool:
            if _context_establishes(fact):
                trace.append(f"fact {fact!r} already in context")
                return True
            if depth >= depth_limit:
                trace.append(f"depth limit reached proving {fact!r}")
                return False
            for rule in sorted(self._rules, key=lambda r: (-r.priority, r.name)):
                if rule.concludes() != fact:
                    continue
                if rule.name in chain:
                    continue
                requirements = rule.requires()
                if not all(_search(r, depth + 1, [*chain, rule.name]) for r in requirements):
                    continue
                if rule.condition(_merged()):
                    chain.append(rule.name)
                    steps.append(rule.name)
                    derived.add(fact)
                    trace.append(f"rule {rule.name!r} v{rule.version} fired -> {fact!r}")
                    return True
            trace.append(f"no rule can establish {fact!r}")
            return False

        established = _search(goal, 0, [])
        return Proof(goal, established, len(steps), steps, trace)

    def prove_all(
        self,
        goals: list[str],
        context: dict[str, Any],
        depth_limit: int = 8,
    ) -> dict[str, Proof]:
        """Backward-chain for every goal in ``goals`` (one :class:`Proof` each)."""
        return {goal: self.prove(goal, context, depth_limit=depth_limit) for goal in goals}

    # ------------------------------------------------------------------ misc

    def _select(self, names: list[str] | None) -> list[Rule]:
        if names is None:
            return list(self._rules)
        selected: list[Rule] = []
        for name in names:
            rule = self.get_rule(name)
            if rule is None:
                raise KeyError(f"Unknown rule {name!r}")
            selected.append(rule)
        return selected


__all__ = [
    "SEVERITY_ORDER",
    "VALID_SEVERITIES",
    "Proof",
    "Rule",
    "RuleEngine",
    "RuleResult",
    "make_rule",
]
