"""tests/agents/test_scheduler.py
================================
Unit tests for cadgenesis.agents.scheduler.
"""

from __future__ import annotations

import pytest

from cadgenesis.agents.scheduler import AgentTask, TaskScheduler


def _task(scheduler, task_id, deps=(), priority=0):
    return scheduler.submit(
        AgentTask(
            task_id=task_id,
            role="r",
            action="a",
            priority=priority,
            depends_on=list(deps),
        )
    )


def test_submit_and_duplicate():
    scheduler = TaskScheduler()
    _task(scheduler, "t1")
    with pytest.raises(ValueError):
        _task(scheduler, "t1")


def test_next_tasks_priority_order():
    scheduler = TaskScheduler()
    _task(scheduler, "low", priority=1)
    _task(scheduler, "high", priority=9)
    tasks = scheduler.next_tasks()
    assert [t.task_id for t in tasks] == ["high", "low"]


def test_next_tasks_respects_dependencies():
    scheduler = TaskScheduler()
    _task(scheduler, "b", deps=["a"])
    _task(scheduler, "a")
    first = scheduler.next_tasks()
    assert [t.task_id for t in first] == ["a"]
    scheduler.mark_completed("a")
    second = scheduler.next_tasks()
    assert [t.task_id for t in second] == ["b"]


def test_status_lifecycle():
    scheduler = TaskScheduler()
    _task(scheduler, "t1")
    scheduler.mark_ready("t1")
    scheduler.mark_running("t1")
    scheduler.mark_completed("t1")
    task = scheduler.get("t1")
    assert task.status == "completed"


def test_step_promotes():
    scheduler = TaskScheduler()
    _task(scheduler, "a")
    _task(scheduler, "b", deps=["a"])
    promoted = scheduler.step()
    assert "a" in [t.task_id for t in promoted]
    assert all(t.status == "ready" for t in scheduler.ready)


def test_progress():
    scheduler = TaskScheduler()
    _task(scheduler, "a")
    _task(scheduler, "b")
    scheduler.mark_completed("a")
    progress = scheduler.progress()
    assert progress["total"] == 2
    assert progress["completed"] == 1
    assert progress["pending"] == 1


def test_has_cycles():
    scheduler = TaskScheduler()
    _task(scheduler, "a", deps=["b"])
    _task(scheduler, "b", deps=["a"])
    assert scheduler.has_cycles()
    scheduler2 = TaskScheduler()
    _task(scheduler2, "a")
    _task(scheduler2, "b", deps=["a"])
    assert not scheduler2.has_cycles()


def test_unknown_task_status():
    scheduler = TaskScheduler()
    with pytest.raises(KeyError):
        scheduler.mark_completed("missing")
