"""tests/continual_learning/test_evaluator.py
============================================
Unit tests for the continual-learning evaluator metrics.
"""

from __future__ import annotations

import pytest

from cadgenesis.continual_learning.evaluator import ContinualEvaluator


def test_record_and_history():
    evaluator = ContinualEvaluator()
    evaluator.record_task_acc("t1", 0.9)
    evaluator.record_task_acc("t1", 0.7)
    assert evaluator.accuracy_history("t1") == [0.9, 0.7]
    assert evaluator.accuracy_history("missing") == []
    assert evaluator.task_ids() == ["t1"]


def test_forgetting_single_measurement_is_zero():
    evaluator = ContinualEvaluator()
    evaluator.record_task_acc("t1", 0.9)
    assert evaluator.forgetting("t1") == 0.0
    assert evaluator.forgetting("unknown") == 0.0


def test_forgetting_best_so_far_minus_latest():
    evaluator = ContinualEvaluator()
    for acc in (0.9, 0.7, 0.8, 0.6):
        evaluator.record_task_acc("t1", acc)
    assert evaluator.forgetting("t1") == pytest.approx(0.9 - 0.6)


def test_negative_forgetting_is_improvement():
    evaluator = ContinualEvaluator()
    evaluator.record_task_acc("t2", 0.5)
    evaluator.record_task_acc("t2", 0.6)
    assert evaluator.forgetting("t2") == pytest.approx(-0.1)


def test_average_forgetting():
    evaluator = ContinualEvaluator()
    evaluator.record_task_acc("t1", 0.9)
    evaluator.record_task_acc("t1", 0.8)  # forgetting 0.1
    evaluator.record_task_acc("t2", 0.7)
    evaluator.record_task_acc("t2", 0.4)  # forgetting 0.3
    assert evaluator.average_forgetting() == pytest.approx(0.2)
    assert ContinualEvaluator().average_forgetting() == 0.0


def test_summary():
    evaluator = ContinualEvaluator()
    evaluator.record_task_acc("t1", 0.9)
    evaluator.record_task_acc("t1", 0.8)
    summary = evaluator.summary()
    assert summary["tasks"] == {"t1": [0.9, 0.8]}
    assert summary["average_forgetting"] == pytest.approx(0.1)
    assert summary["num_tasks"] == 1
    assert summary["total_measurements"] == 2
