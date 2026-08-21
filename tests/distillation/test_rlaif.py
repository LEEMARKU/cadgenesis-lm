"""tests/distillation/test_rlaif.py"""

from __future__ import annotations

import math

import pytest
import torch

from cadgenesis.distillation.rlaif import RLAIFEngine


def sample(toon: str, score: float) -> dict:
    return {"toon": toon, "score": score}


def test_preference_pairs_orders_by_score_descending():
    engine = RLAIFEngine()
    pairs = engine.preference_pairs([sample("c", 0.9), sample("a", 0.3), sample("b", 0.6)])
    assert pairs == [(sample("c", 0.9), sample("b", 0.6)), (sample("b", 0.6), sample("a", 0.3))]
    for chosen, rejected in pairs:
        assert chosen["score"] > rejected["score"]


def test_preference_pairs_drops_equal_score_ties():
    engine = RLAIFEngine()
    pairs = engine.preference_pairs([sample("a", 0.5), sample("b", 0.5), sample("c", 0.9)])
    assert pairs == [(sample("c", 0.9), sample("a", 0.5))]
    assert len(pairs) == 1


def test_preference_pairs_skips_entries_without_score():
    engine = RLAIFEngine()
    pairs = engine.preference_pairs([sample("a", 0.9), {"toon": "b"}, {"score": 0.5}])
    assert len(pairs) == 0


def test_preference_pairs_empty_input():
    assert RLAIFEngine().preference_pairs([]) == []


def test_bt_loss_lower_when_chosen_exceeds_rejected():
    engine = RLAIFEngine()
    chosen_higher = torch.tensor([1.0, 0.5])
    rejected_lower = torch.tensor([-1.0, -0.5])
    loss_good = engine.bradley_terry_loss(chosen_higher, rejected_lower)
    loss_bad = engine.bradley_terry_loss(rejected_lower, chosen_higher)
    assert loss_good.item() < loss_bad.item()
    assert loss_good.ndim == 0


def test_bt_loss_matches_manual_logsigmoid():
    chosen = torch.tensor([1.5, -0.2])
    rejected = torch.tensor([-0.5, 0.3])
    loss = RLAIFEngine().bradley_terry_loss(chosen, rejected)
    manual = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_bt_loss_zero_at_equal_logits():
    logits = torch.randn(3)
    loss = RLAIFEngine().bradley_terry_loss(logits, logits.clone())
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_bt_loss_label_smoothing_changes_loss():
    chosen = torch.tensor([2.0])
    rejected = torch.tensor([-2.0])
    plain = RLAIFEngine().bradley_terry_loss(chosen, rejected)
    smoothed = RLAIFEngine(label_smoothing=0.1).bradley_terry_loss(chosen, rejected)
    assert torch.isfinite(smoothed)
    assert abs(plain.item() - smoothed.item()) > 1e-6


def test_bt_loss_per_call_override():
    chosen = torch.tensor([1.0])
    rejected = torch.tensor([-1.0])
    default = RLAIFEngine().bradley_terry_loss(chosen, rejected)
    overridden = RLAIFEngine().bradley_terry_loss(chosen, rejected, label_smoothing=0.2)
    assert abs(default.item() - overridden.item()) > 1e-6


def test_rejects_invalid_label_smoothing():
    with pytest.raises(ValueError):
        RLAIFEngine(label_smoothing=0.7)
    with pytest.raises(ValueError):
        RLAIFEngine().bradley_terry_loss(torch.tensor([1.0]), torch.tensor([0.0]), 0.5)


def test_reward_from_critiques_mapping():
    engine = RLAIFEngine()
    rewards = engine.reward_from_critiques(
        [{"toon": "a", "score": 0.9}, {"toon": "b", "score": 0.2}, {"score": 1.0}]
    )
    assert rewards == {"a": 0.9, "b": 0.2}


def test_reward_from_critiques_clamps_scores():
    rewards = RLAIFEngine().reward_from_critiques(
        [{"toon": "a", "score": 5.0}, {"toon": "b", "score": -1.0}]
    )
    assert rewards == {"a": 1.0, "b": 0.0}
