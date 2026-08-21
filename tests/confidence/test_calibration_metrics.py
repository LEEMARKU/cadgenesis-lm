"""
tests/confidence/test_calibration_metrics.py
============================================
Tests for Brier score, reliability diagram and abstention policy
(pre-training gate: confidence-calibration infrastructure).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.confidence.calibration import brier_score, reliability_diagram
from cadgenesis.confidence.fallback import AbstentionPolicy, FallbackStrategy


def _random_probs(n: int = 200, c: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    logits = torch.randn(n, c)
    probs = torch.softmax(logits, dim=-1)
    labels = torch.randint(0, c, (n,))
    return probs, labels


class TestBrierScore:
    def test_perfect_predictions_zero(self):
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1, 1])
        assert brier_score(probs, labels) == pytest.approx(0.0)

    def test_worst_case_two_classes(self):
        probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        labels = torch.tensor([1, 1])
        # per-sample: (1-0)^2 + (0-1)^2 = 2.0
        assert brier_score(probs, labels) == pytest.approx(2.0)

    def test_uniform_probs(self):
        probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        labels = torch.tensor([0, 0])
        # per-sample: (0.5-1)^2 + (0.5-0)^2 = 0.5
        assert brier_score(probs, labels) == pytest.approx(0.5)

    def test_flattened_3d(self):
        probs = torch.randn(4, 3, 5).softmax(dim=-1)
        labels = torch.randint(0, 5, (4, 3))
        flat = brier_score(probs, labels)
        stacked = torch.stack([p for p in probs])
        labels_flat = labels.reshape(-1)
        assert flat == pytest.approx(brier_score(stacked, labels_flat))

    def test_empty_input(self):
        assert brier_score(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)) == 0.0

    def test_range(self):
        probs, labels = _random_probs()
        score = brier_score(probs, labels)
        assert 0.0 <= score <= 1.0


class TestReliabilityDiagram:
    def test_perfectly_calibrated(self):
        probs, labels = _random_probs(seed=1)
        diagram = reliability_diagram(probs, labels, n_bins=10)
        assert diagram["n_bins"] == 10
        assert len(diagram["points"]) > 0
        assert diagram["ece"] >= 0.0

    def test_points_schema(self):
        probs, labels = _random_probs(seed=2)
        diagram = reliability_diagram(probs, labels, n_bins=5)
        for point in diagram["points"]:
            assert set(point) >= {
                "bin_index",
                "bin_lower",
                "bin_upper",
                "confidence",
                "accuracy",
                "count",
            }
            assert 0.0 <= point["confidence"] <= 1.0
            assert 0.0 <= point["accuracy"] <= 1.0

    def test_confidence_calibrated_data_lower_ece(self):
        torch.manual_seed(3)
        logits = torch.randn(500, 8) * 10.0
        labels = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        # Sharply confident + always correct: calibration should be near-perfect
        diagram = reliability_diagram(probs, labels, n_bins=10)
        assert diagram["ece"] < 0.15

    def test_3d_input(self):
        probs = torch.randn(10, 4, 6).softmax(dim=-1)
        labels = torch.randint(0, 6, (10, 4))
        diagram = reliability_diagram(probs, labels, n_bins=4)
        assert diagram["points"]


class TestAbstentionPolicy:
    def test_should_abstain_below_threshold(self):
        policy = AbstentionPolicy(threshold=0.6)
        assert policy.should_abstain(0.4) is True
        assert policy.should_abstain(0.6) is False

    def test_should_abstain_on_uncertainty(self):
        policy = AbstentionPolicy(threshold=0.1, max_uncertainty=0.7)
        assert policy.should_abstain(0.9, uncertainty=0.95) is True
        assert policy.should_abstain(0.9, uncertainty=0.5) is False

    def test_decide_strategy(self):
        policy = AbstentionPolicy(threshold=0.6)
        decision = policy.decide(0.3)
        assert decision["strategy"] == FallbackStrategy.ABSTAIN
        decision = policy.decide(0.9)
        assert decision["strategy"] != FallbackStrategy.ABSTAIN

    def test_abstention_rate(self):
        policy = AbstentionPolicy(threshold=0.5)
        assert policy.abstention_rate([0.9, 0.8, 0.4, 0.1]) == pytest.approx(0.5)
        assert policy.abstention_rate([]) == 0.0

    def test_selective_accuracy(self):
        policy = AbstentionPolicy(threshold=0.5)
        confidences = [0.9, 0.8, 0.4, 0.2]
        correct = [True, False, True, False]
        # accepted: 0.9 (correct), 0.8 (wrong) -> 0.5
        assert policy.selective_accuracy(confidences, correct) == pytest.approx(0.5)
        assert policy.selective_accuracy([0.1], [True]) == 0.0