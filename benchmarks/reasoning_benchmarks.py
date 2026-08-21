"""
benchmarks/reasoning_benchmarks.py
==================================
Benchmarks for the Pillar 7 (v6.0) neuro-symbolic reasoning stack.

Measures wall-clock time for:
* rule engine (forward chaining + backward chaining + version snapshot),
* knowledge graph (build / search / shortest path),
* constraint solver (solve / propagate / conflict detection / repair),
* engineering standards (lookups + compliance over a part corpus),
* symbolic planner (planning / decomposition),
* hybrid pipeline (full neuro-symbolic run with all stages active).

Run with::

    python benchmarks/reasoning_benchmarks.py                    # all
    python benchmarks/reasoning_benchmarks.py --sections hybrid
"""

from __future__ import annotations

import argparse
import time

from cadgenesis.reasoning.constraint_solver import Constraint, ConstraintSolver, Variable
from cadgenesis.reasoning.hybrid import HybridReasoningPipeline
from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph
from cadgenesis.reasoning.rule_engine import Rule, RuleEngine, make_rule
from cadgenesis.reasoning.standards import default_standards_library
from cadgenesis.reasoning.symbolic_planner import (
    PlanningOperator,
    SymbolicPlanner,
)

SECTIONS = (
    "rules",
    "graph",
    "constraints",
    "standards",
    "planner",
    "hybrid",
)


def time_ms(fn, reps: int) -> str:
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return f"{sum(times) / len(times) * 1000:8.3f} ms"


def report(label: str, fn, reps: int) -> None:
    print(f"{label:>12} | {time_ms(fn, reps)}")


def _build_rules() -> RuleEngine:
    engine = RuleEngine()
    engine.add_rules(
        [
            make_rule(
                "wall_ok",
                lambda ctx: ctx.get("min_wall", 0.0) >= 0.8,
                severity="error",
                meta={"concludes": "wall_ok"},
            ),
            make_rule(
                "draft_ok",
                lambda ctx: ctx.get("draft_angle", 0.0) >= 1.0,
                severity="warning",
                meta={"concludes": "draft_ok", "requires": ["wall_ok"]},
            ),
            make_rule(
                "design_ok",
                lambda ctx: bool(ctx.get("wall_ok")),
                severity="info",
                meta={"concludes": "design_ok", "requires": ["wall_ok", "draft_ok"]},
            ),
            Rule(
                "material_ok",
                condition=lambda ctx: "material" in ctx,
                action=lambda ctx: ctx.update(material_ok=True) or "material ok",
                severity="info",
                priority=5,
            ),
        ]
    )
    return engine


def _build_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for i in range(100):
        graph.add_node(f"concept{i}", label=f"concept {i}", node_type="concept")
    for i in range(99):
        graph.add_edge(f"concept{i}", f"concept{i + 1}", "links")
    return graph


def _build_solver() -> tuple[ConstraintSolver, list[Variable], list[Constraint]]:
    solver = ConstraintSolver()
    variables = [Variable(f"v{i}", initial=1.0, lower=0.0, upper=10.0) for i in range(8)]
    constraints = [
        Constraint(f"c{i}", {f"v{i}": 1.0, f"v{(i + 1) % 8}": 1.0}, "==", 2.0) for i in range(8)
    ]
    return solver, variables, constraints


def bench_rules(reps: int) -> None:
    engine = _build_rules()
    context = {"min_wall": 1.0, "draft_angle": 2.0, "material": "steel"}
    report("add-rule", lambda: engine.add_rules([]), reps)
    report("evaluate", lambda: engine.evaluate(context), reps)
    report("forward-run", lambda: engine.run(context), reps)
    report("backward-prove", lambda: engine.prove("design_ok", context), reps)
    report("snapshot", lambda: engine.snapshot(), reps)


def bench_graph(reps: int) -> None:
    graph = _build_graph()
    report("search", lambda: graph.find_related("concept0", max_depth=3), reps)
    report("shortest-path", lambda: graph.shortest_path("concept0", "concept90"), reps)
    report("to-json", lambda: graph.to_json(), reps)


def bench_constraints(reps: int) -> None:
    solver, variables, constraints = _build_solver()
    report("solve", lambda: solver.solve(variables, constraints), reps)
    report("propagate", lambda: solver.propagate(variables, constraints, {}), reps)
    report("conflicts", lambda: solver.detect_conflicts(variables, constraints), reps)
    report("repair", lambda: solver.repair(variables, constraints), reps)


def bench_standards(reps: int) -> None:
    library = default_standards_library()
    part = {"kind": "tolerance", "grade": 7, "tolerance_um": 25.0, "standards": ["ISO"]}
    report("tolerance-lookup", lambda: library.tolerance(50.0, 7), reps)
    report("compliance", lambda: library.compliance(part), reps)
    report("summary", lambda: library.summary(), reps)


def bench_planner(reps: int) -> None:
    planner = SymbolicPlanner(
        [
            PlanningOperator(
                "sketch",
                precondition=lambda s: not bool(s.get("sketched")),
                effect=lambda s: {**s, "sketched": True},
                action="sketch",
            ),
            PlanningOperator(
                "model",
                precondition=lambda s: bool(s.get("sketched")) and not bool(s.get("modeled")),
                effect=lambda s: {**s, "modeled": True},
                action="model",
            ),
            PlanningOperator(
                "validate",
                precondition=lambda s: bool(s.get("modeled")) and not bool(s.get("validated")),
                effect=lambda s: {**s, "validated": True},
                action="validate",
            ),
        ]
    )
    report(
        "plan",
        lambda: planner.plan("build", {}, lambda s: bool(s.get("validated"))),
        reps,
    )
    report(
        "decompose",
        lambda: planner.decompose("build", lambda s: bool(s.get("validated")), {}),
        reps,
    )


def bench_hybrid(reps: int) -> None:
    pipeline = HybridReasoningPipeline(rule_engine=_build_rules())
    context = {
        "query": "ISO 286",
        "min_wall": 1.0,
        "draft_angle": 2.0,
        "material": "steel",
        "part": {
            "processes": ["machining"],
            "min_wall_thickness": 1.0,
            "hole_diameter": 4.0,
            "hole_depth": 10.0,
        },
    }
    report("hybrid-reason", lambda: pipeline.reason(context), reps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    for section in args.sections:
        print(f"\n== {section} ==")
        if section == "rules":
            bench_rules(args.reps)
        elif section == "graph":
            bench_graph(args.reps)
        elif section == "constraints":
            bench_constraints(args.reps)
        elif section == "standards":
            bench_standards(args.reps)
        elif section == "planner":
            bench_planner(args.reps)
        elif section == "hybrid":
            bench_hybrid(args.reps)


if __name__ == "__main__":
    main()
