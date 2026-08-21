"""tests/distillation/test_consensus.py"""

from __future__ import annotations

import torch

from cadgenesis.distillation.consensus import TeacherConsensus

TOON_A = "id|feature\n1|BOX"
TOON_B = "id|feature\n1|CYLINDER"


def test_toon_votes_majority_winner_and_agreement():
    outputs = {"t1": TOON_A, "t2": TOON_A, "t3": TOON_B}
    result = TeacherConsensus().toon_votes(outputs)
    assert result.consensus_toon == TOON_A
    assert result.agreement_score == 2 / 3
    assert result.vote_counts == {TOON_A: 2, TOON_B: 1}
    assert result.winner_teachers == ["t1", "t2"]


def test_toon_votes_tie_broken_by_first_seen_stable_order():
    outputs = {"b": TOON_A, "a": TOON_B}
    result = TeacherConsensus().toon_votes(outputs)
    assert result.consensus_toon == TOON_B
    assert result.agreement_score == 0.5
    assert result.winner_teachers == ["a"]


def test_toon_votes_custom_stable_order_breaks_tie():
    outputs = {"b": TOON_A, "a": TOON_B}
    result = TeacherConsensus(stable_order=["b", "a"]).toon_votes(outputs)
    assert result.consensus_toon == TOON_A
    assert result.winner_teachers == ["b"]


def test_toon_votes_weights_change_winner():
    outputs = {"t1": TOON_A, "t2": TOON_A, "t3": TOON_B}
    weights = {"t1": 0.4, "t2": 0.4, "t3": 2.0}
    result = TeacherConsensus().toon_votes(outputs, weights)
    assert result.consensus_toon == TOON_B
    assert result.agreement_score == 2.0 / 2.8
    assert result.vote_counts == {TOON_A: 2, TOON_B: 1}


def test_toon_votes_empty_outputs():
    result = TeacherConsensus().toon_votes({})
    assert result.consensus_toon == ""
    assert result.agreement_score == 0.0
    assert result.vote_counts == {}
    assert result.winner_teachers == []


def test_sequence_consensus_delegates_to_engine():
    result = TeacherConsensus().sequence_consensus(
        {"t1": ["BOX", "EXTRUDE"], "t2": ["BOX", "EXTRUDE"], "t3": ["CYLINDER"]}
    )
    assert result == (["BOX", "EXTRUDE"], 2 / 3)


def test_consensus_logits_mean_and_full_agreement():
    torch.manual_seed(7)
    logits = torch.randn(3, 2, 4, 5)
    mean_logits, agreement = TeacherConsensus.consensus_logits(logits)
    assert mean_logits.shape == (2, 4, 5)
    assert torch.allclose(mean_logits, logits.mean(dim=0))
    assert 0.0 <= agreement <= 1.0


def test_consensus_logits_identical_teachers_full_agreement():
    logits = torch.randn(2, 4, 5)
    stacked = logits.unsqueeze(0).repeat(4, 1, 1, 1)
    mean_logits, agreement = TeacherConsensus.consensus_logits(stacked)
    assert torch.allclose(mean_logits, logits)
    assert agreement == 1.0


def test_consensus_logits_disagreeing_teachers_low_agreement():
    a = torch.zeros(1, 1, 4)
    b = torch.zeros(1, 1, 4)
    a[0, 0, 0] = 10.0
    b[0, 0, 1] = 10.0
    stacked = torch.stack([a, b])
    _, agreement = TeacherConsensus.consensus_logits(stacked)
    assert agreement < 1.0


def test_consensus_logits_rejects_non_4d():
    with __import__("pytest").raises(ValueError):
        TeacherConsensus.consensus_logits(torch.randn(3, 4, 5))
