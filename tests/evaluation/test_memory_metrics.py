"""tests/evaluation/test_memory_metrics.py
=========================================
Unit tests for the Pillar 6 memory metrics.
"""

from __future__ import annotations

import pytest

from cadgenesis.evaluation.memory_metrics import (
    compression_ratio,
    consolidation_ratio,
    mean_reciprocal_rank,
    memory_hit_rate,
    precision_at_k,
    recall_at_k,
    routing_accuracy,
    run_memory_benchmark,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b"], ["a", "c", "b"], k=2) == pytest.approx(0.5)
    assert precision_at_k(["a"], ["a", "b"], k=1) == 1.0
    assert precision_at_k(["a"], []) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "c"]) == pytest.approx(0.5)
    assert recall_at_k(["a"], ["a"], k=1) == 1.0
    assert recall_at_k([], ["a"]) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(["b"], ["a", "b", "c"]) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["z"], ["a", "b"]) == 0.0


def test_routing_accuracy():
    assert routing_accuracy(["cad", "eng"], ["cad", "eng"]) == 1.0
    assert routing_accuracy(["cad", "eng"], ["cad", "sim"]) == pytest.approx(0.5)
    assert routing_accuracy([], []) == 0.0


def test_memory_hit_rate():
    assert memory_hit_rate(["a", "b", "c"], ["a", "c"]) == pytest.approx(2 / 3)
    assert memory_hit_rate([], []) == 0.0


def test_consolidation_ratio():
    assert consolidation_ratio(4, 1) == 0.25
    assert consolidation_ratio(0, 1) == 0.0


def test_compression_ratio():
    assert compression_ratio(100, 25) == 4.0
    assert compression_ratio(10, 0) == 0.0


def test_run_memory_benchmark():
    report = run_memory_benchmark(
        retrieval_batches=[(["a", "b"], ["a", "c", "b"]), (["x"], ["x", "y"])],
        routing_preds=[("cad", "cad"), ("sim", "sim")],
        consolidation_batch=(4, 1),
        compression_batch=(100, 25),
    )
    assert report["routing_accuracy"] == 1.0
    assert report["consolidation_ratio"] == 0.25
    assert report["compression_ratio"] == 4.0
    assert report["checks"]["retrieval_batches"] == 2
    assert report["checks"]["routing_queries"] == 2


def test_run_memory_benchmark_empty():
    report = run_memory_benchmark([])
    assert report["routing_accuracy"] == 0.0
    assert report["consolidation_ratio"] == 0.0
