"""cadgenesis.reasoning.constraint_solver
========================================
Numeric constraint solver for 2D/3D geometric and dimensional consistency.

Variables are real-valued with optional hard bounds; constraints are linear
combinations ``sum(coef_i * value_i) op rhs``.  A projection-based iterative
solver finds a consistent assignment (when one exists) by repeatedly projecting
each violated constraint onto its feasible set while clamping to variable
bounds.

Typical CAD use: check that a dimensional assignment (width, height, depth,
diameters, thicknesses) satisfies design equations and inequalities such as
``width + 2 * wall >= outer_width`` or ``thickness <= diameter / 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_VALID_OPERATORS = ("==", "<=", ">=")


@dataclass
class Variable:
    """A bounded real-valued design variable."""

    name: str
    initial: float = 0.0
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variable name must be non-empty")
        if not math.isfinite(self.initial):
            raise ValueError(f"initial must be finite, got {self.initial!r}")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"empty bound range for {self.name!r}: [{self.lower}, {self.upper}]")
        if self.lower is not None and self.initial < self.lower:
            self.initial = self.lower
        if self.upper is not None and self.initial > self.upper:
            self.initial = self.upper

    def clamp(self, value: float) -> float:
        if self.lower is not None and value < self.lower:
            return self.lower
        if self.upper is not None and value > self.upper:
            return self.upper
        return value

    def feasible_span(self) -> float:
        lo = self.lower if self.lower is not None else -math.inf
        hi = self.upper if self.upper is not None else math.inf
        if math.isinf(lo) or math.isinf(hi):
            return math.inf
        return hi - lo


@dataclass
class Constraint:
    """A linear constraint ``sum(terms[name] * name) op rhs``."""

    name: str
    terms: dict[str, float]
    operator: str
    rhs: float
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Constraint name must be non-empty")
        if self.operator not in _VALID_OPERATORS:
            raise ValueError(
                f"invalid operator {self.operator!r}; expected one of {_VALID_OPERATORS}"
            )
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not self.terms:
            raise ValueError("Constraint must reference at least one variable")

    def value(self, assignment: dict[str, float]) -> float:
        return sum(coef * assignment.get(name, 0.0) for name, coef in self.terms.items())

    def residual(self, assignment: dict[str, float]) -> float:
        """Signed violation magnitude.

        Zero (or negative for inequalities) means satisfied; for ``==`` a
        positive or negative value both indicate violation magnitude.
        """
        value = self.value(assignment)
        if self.operator == "==":
            return value - self.rhs
        if self.operator == "<=":
            return value - self.rhs
        return self.rhs - value

    def satisfied(self, assignment: dict[str, float]) -> bool:
        if self.operator == "==":
            return abs(self.residual(assignment)) <= self.tolerance
        return self.residual(assignment) <= self.tolerance


@dataclass
class Solution:
    """Result of solving a constraint system."""

    feasible: bool
    assignment: dict[str, float]
    iterations: int
    residuals: dict[str, float]
    max_residual: float
    messages: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        violated = sum(1 for r in self.residuals.values() if abs(r) > 1e-6)
        return {
            "feasible": self.feasible,
            "iterations": self.iterations,
            "max_residual": self.max_residual,
            "satisfied": len(self.residuals) - violated,
            "violated": violated,
        }


class ConstraintSolver:
    """Projection-based solver for linear equality/inequality systems."""

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> Solution:
        """Find a feasible assignment for ``variables`` satisfying ``constraints``.

        Bounds are enforced throughout; a constraint that fights the bounds
        yields an infeasible solution with a diagnostic message.
        """
        if not variables:
            return Solution(True, {}, 0, {}, 0.0)
        if not constraints:
            assignment = {v.name: v.initial for v in variables}
            return Solution(True, assignment, 0, {}, 0.0)

        variable_map = {v.name: v for v in variables}
        constraints_by_name = {c.name: c for c in constraints}
        for constraint in constraints:
            unknown = [n for n in constraint.terms if n not in variable_map]
            if unknown:
                raise KeyError(
                    f"Constraint {constraint.name!r} references unknown variables: {unknown}"
                )

        assignment = {v.name: float(v.initial) for v in variables}
        messages: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            worst = 0.0
            for constraint in constraints:
                if constraint.satisfied(assignment):
                    continue
                residual = constraint.residual(assignment)
                worst = max(worst, abs(residual))
                # Project the linear combination toward its target value.
                delta_needed = constraint.value(assignment) - constraint.rhs
                denom = sum(coef * coef for coef in constraint.terms.values() if coef != 0.0)
                if denom == 0.0:
                    continue
                for name, coef in constraint.terms.items():
                    if coef == 0.0:
                        continue
                    var = variable_map[name]
                    if var.upper is not None and var.lower is not None and var.upper == var.lower:
                        continue
                    correction = (delta_needed * coef) / denom
                    assignment[name] = var.clamp(assignment[name] - correction)
            if worst <= self.tolerance:
                residuals = {c.name: c.residual(assignment) for c in constraints}
                return Solution(True, assignment, iteration, residuals, worst, messages)

        residuals = {c.name: c.residual(assignment) for c in constraints}
        worst = max((abs(r) for r in residuals.values()), default=0.0)
        for name, residual in residuals.items():
            if not constraints_by_name[name].satisfied(assignment):
                messages.append(
                    f"Constraint {name!r} could not be satisfied "
                    f"(residual {residual:.2e}); check variable bounds."
                )
        return Solution(False, assignment, self.max_iterations, residuals, worst, messages)

    def check_consistency(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> bool:
        """Convenience wrapper returning feasibility only."""
        return self.solve(variables, constraints).feasible

    # ------------------------------------------------------ P7 extensions

    def dependency_graph(
        self,
        constraints: list[Constraint],
    ) -> dict[str, list[str]]:
        """Constraints that share a variable (dependency propagation edges).

        ``A -> [B, C]`` means A and B/C share at least one variable, so a
        change propagated through A must be re-checked against B and C.
        """
        graph: dict[str, list[str]] = {c.name: [] for c in constraints}
        for i, left in enumerate(constraints):
            for right in constraints[i + 1 :]:
                if set(left.terms) & set(right.terms):
                    graph[left.name].append(right.name)
                    graph[right.name].append(left.name)
        return graph

    def propagate(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        assignment: dict[str, float],
        max_hops: int = 4,
    ) -> dict[str, float]:
        """Propagate an assignment through constraint dependencies.

        When a variable value changes, every connected constraint is re-checked
        and re-projected (in dependency order) until the assignment stabilises
        or ``max_hops`` rounds elapse.  Returns the propagated assignment.
        """
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        variable_map = {v.name: v for v in variables}
        work = dict(assignment)
        for variable in variables:
            work.setdefault(variable.name, variable.initial)
        for _ in range(max_hops):
            changed = False
            for constraint in constraints:
                if constraint.satisfied(work):
                    continue
                delta_needed = constraint.value(work) - constraint.rhs
                denom = sum(c * c for c in constraint.terms.values() if c != 0.0)
                if denom == 0.0:
                    continue
                for name, coef in constraint.terms.items():
                    var = variable_map.get(name)
                    if var is None or coef == 0.0:
                        continue
                    if var.upper is not None and var.lower is not None and var.upper == var.lower:
                        continue
                    updated = var.clamp(work[name] - (delta_needed * coef) / denom)
                    if abs(updated - work[name]) > 1e-12:
                        work[name] = updated
                        changed = True
            if not changed:
                break
        return work

    def detect_conflicts(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> list[dict[str, Any]]:
        """Report conflicting constraint pairs after an attempted solve.

        Two constraints that share a variable conflict when the system is
        infeasible but dropping *either one* alone restores feasibility — i.e.
        the pair jointly over-constrains the shared variable.  Each entry
        carries ``left``/``right`` constraint names and an explanation.
        """
        if self.solve(variables, constraints).feasible:
            return []
        variable_map = {v.name: v for v in variables}
        pairs: list[tuple[Constraint, Constraint]] = []
        for constraint in constraints:
            for name in constraint.terms:
                if name not in variable_map:
                    continue
                for other in constraints:
                    if other.name == constraint.name or name not in other.terms:
                        continue
                    key = tuple(sorted((constraint.name, other.name)))
                    if key not in {tuple(sorted((a.name, b.name))) for a, b in pairs}:
                        pairs.append((constraint, other))
        conflicts: list[dict[str, Any]] = []
        for left, right in pairs:
            without_left = [c for c in constraints if c.name != left.name]
            without_right = [c for c in constraints if c.name != right.name]
            if (
                self.solve(variables, without_left).feasible
                or self.solve(variables, without_right).feasible
            ):
                shared = sorted(set(left.terms) & set(right.terms))
                conflicts.append(
                    {
                        "left": left.name,
                        "right": right.name,
                        "variable": shared[0] if shared else "",
                        "detail": (
                            f"{left.name!r} and {right.name!r} jointly "
                            f"over-constrain {shared}; dropping either "
                            f"restores feasibility"
                        ),
                    }
                )
        return conflicts

    def repair(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        relax_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Automatically repair an infeasible system.

        Drops the lowest-priority conflicting constraint (``relax_order`` gives
        the drop order; otherwise constraints are dropped by residual size) and
        re-solves until feasible or no constraints remain.  Returns a report
        with ``feasible``, ``dropped`` (constraint names), ``assignment`` and
        ``messages``.
        """
        remaining = list(constraints)
        dropped: list[str] = []
        while True:
            solution = self.solve(variables, remaining)
            if solution.feasible:
                return {
                    "feasible": True,
                    "dropped": dropped,
                    "assignment": solution.assignment,
                    "messages": list(solution.messages),
                }
            if not remaining:
                return {
                    "feasible": False,
                    "dropped": dropped,
                    "assignment": solution.assignment,
                    "messages": [
                        *list(solution.messages),
                        "no constraints left to relax",
                    ],
                }
            unsolved = [c for c in remaining if not c.satisfied(solution.assignment)]
            if relax_order is not None:
                victim = next(
                    (c for c in unsolved if c.name in relax_order),
                    unsolved[0],
                )
            else:
                victim = max(unsolved, key=lambda c: abs(c.residual(solution.assignment)))
            remaining.remove(victim)
            dropped.append(victim.name)


__all__ = [
    "Constraint",
    "ConstraintSolver",
    "Solution",
    "Variable",
]
