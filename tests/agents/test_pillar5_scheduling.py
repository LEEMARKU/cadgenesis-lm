"""tests/agents/test_pillar5_scheduling.py
==========================================
Unit tests for the Pillar 5 scheduling layer (DAG scheduler, priorities,
deadlines, dynamic scheduling, load balancing).
"""

from __future__ import annotations

import time

import pytest

from cadgenesis.agents.scheduling import (
    DAGScheduler,
    DeadlineScheduler,
    DynamicScheduler,
    LoadBalancer,
    PriorityScheduler,
    TaskGraph,
    TaskNode,
)


def _node(task_id, depends_on=(), priority=0, cost=1.0, deadline=None):
    return TaskNode(
        task_id=task_id,
        role="r",
        action="a",
        depends_on=list(depends_on),
        priority=priority,
        cost=cost,
        deadline=deadline,
    )


# ------------------------------------------------------------------ task graph


def test_graph_add_unknown_dependency_raises():
    graph = TaskGraph()
    with pytest.raises(ValueError):
        graph.add(_node("b", depends_on=["a"]))


def test_graph_cycle_detection():
    graph = TaskGraph()
    graph.add(_node("b"))
    graph.add(_node("a", depends_on=["b"]))
    graph.add_dependency("b", "a")
    assert not graph.is_acyclic()
    with pytest.raises(ValueError):
        graph.topological_order()


def test_graph_topological_order():
    graph = TaskGraph()
    graph.add(_node("a"))
    graph.add(_node("b", depends_on=["a"]))
    graph.add(_node("c", depends_on=["a"]))
    assert graph.topological_order()[0] == "a"
    assert graph.is_acyclic()


def test_graph_critical_path():
    graph = TaskGraph()
    graph.add(_node("a", cost=1.0))
    graph.add(_node("b", depends_on=["a"], cost=5.0))
    graph.add(_node("c", depends_on=["b"], cost=2.0))
    graph.add(_node("d", depends_on=["a"], cost=1.0))
    path, total = graph.critical_path()
    assert path == ["a", "b", "c"]
    assert total == 8.0


def test_ready_nodes_require_completed_deps():
    graph = TaskGraph()
    graph.add(_node("a"))
    graph.add(_node("b", depends_on=["a"]))
    assert [n.task_id for n in graph.ready_nodes()] == ["a"]
    graph.get("a").status = "completed"
    assert [n.task_id for n in graph.ready_nodes()] == ["b"]


def test_task_node_validation():
    with pytest.raises(ValueError):
        TaskNode(task_id="", role="r", action="a")
    assert _node("x").to_dict()["task_id"] == "x"


# ------------------------------------------------------------------ DAGScheduler


def test_dag_scheduler_executes_in_dependency_order():
    graph = TaskGraph()
    graph.add(_node("a"))
    graph.add(_node("b", depends_on=["a"]))
    graph.add(_node("c", depends_on=["b"]))
    order: list[str] = []

    def execute(node: TaskNode):
        order.append(node.task_id)
        node.result = {"ok": True}
        return node.result

    scheduler = DAGScheduler(workers=2)
    stats = scheduler.run(graph, execute_fn=execute)
    assert order.index("a") < order.index("b") < order.index("c")
    assert stats.completed == 3
    assert stats.total == 3
    scheduler.shutdown()


def test_dag_scheduler_marks_failures():
    graph = TaskGraph()
    graph.add(_node("a"))

    def execute(node: TaskNode):
        node.result = None
        return type("R", (), {"ok": False, "message": "nope"})()

    scheduler = DAGScheduler(workers=2)
    stats = scheduler.run(graph, execute_fn=execute)
    assert stats.failed == 1
    scheduler.shutdown()


def test_dag_scheduler_retries():
    graph = TaskGraph()
    graph.add(_node("a"))
    attempts = {"n": 0}

    def execute(node: TaskNode):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return type("R", (), {"ok": False, "message": "retry me"})()
        return {"ok": True}

    scheduler = DAGScheduler(workers=2, default_retries=2)
    stats = scheduler.run(graph, execute_fn=execute)
    assert stats.completed == 1
    assert graph.get("a").attempts == 2
    scheduler.shutdown()


def test_dag_scheduler_cyclic_graph_raises():
    graph = TaskGraph()
    graph.add(_node("b"))
    graph.add(_node("a", depends_on=["b"]))
    graph.add_dependency("b", "a")
    scheduler = DAGScheduler()
    with pytest.raises(ValueError):
        scheduler.run(graph)
    scheduler.shutdown()


# ----------------------------------------------------------- policy schedulers


def test_priority_scheduler_picks_highest():
    scheduler = PriorityScheduler()
    scheduler.submit(_node("low", priority=1))
    scheduler.submit(_node("high", priority=9))
    assert scheduler.next_task().task_id == "high"
    assert scheduler.progress()["pending"] == 2


def test_deadline_scheduler_edf_and_expiry():
    scheduler = DeadlineScheduler()
    scheduler.submit(_node("soon", deadline=time.time() + 60))
    scheduler.submit(_node("late", deadline=time.time() + 120))
    assert scheduler.next_task().task_id == "soon"
    scheduler.submit(_node("expired", deadline=time.time() - 1))
    late = scheduler.next_task()
    assert late.task_id == "late"
    assert scheduler.pending == 0


def test_dynamic_scheduler_admits_while_running():
    scheduler = DynamicScheduler(workers=1)
    scheduler.submit(_node("t1"))
    scheduler.submit(_node("t2"))
    stats = scheduler.run(execute_fn=lambda n: {"ok": True})
    assert stats.completed == 2
    assert not scheduler.is_running
    scheduler.shutdown()


# ----------------------------------------------------------------- load balance


def test_load_balancer_balances_by_cost():
    balancer = LoadBalancer(worker_count=3)
    assignment = balancer.assign(
        [_node("a", cost=10.0), _node("b", cost=2.0), _node("c", cost=3.0)]
    )
    total = sum(len(v) for v in assignment.values())
    assert total == 3
    assert sorted(balancer.loads()) == sorted([10.0, 2.0, 3.0])
