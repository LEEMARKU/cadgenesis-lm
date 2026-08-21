"""
tests/inference/test_eagle.py
=============================
Tests for EAGLE-style speculative decoding: greedy-preserving verification.
"""

from __future__ import annotations

import torch

from cadgenesis.config import CADConfig
from cadgenesis.inference.eagle import (
    EagleDraftHead,
    collect_hidden_pairs,
    speculative_eagle,
    train_eagle,
)
from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _make_env(seed: int = 0):
    torch.manual_seed(seed)
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box"])
    cfg = CADConfig.mini()
    model = GeometryAwareTransformer(cfg)
    engine = CADInferenceEngine(model, tok, device="cpu")

    def type_of(i: int) -> int:
        try:
            return tok.vocab.type_id_of(int(i))
        except KeyError:
            return 0

    head = EagleDraftHead(d_model=model.d_model, num_heads=4, vocab_size=model.cad_vocab_size)
    seq = [tok.vocab["SKETCH_RECT"], tok.vocab["EXTRUDE"], tok.vocab["BOX"], tok.eos_id]
    pairs = collect_hidden_pairs(model, [seq], type_of)
    return tok, model, engine, type_of, head, pairs


class TestEagleDraftHead:
    def test_predict_next_shapes(self):
        torch.manual_seed(0)
        head = EagleDraftHead(d_model=32, num_heads=4, vocab_size=100)
        hidden = torch.randn(2, 6, 32)
        logits, nh = head.predict_next(hidden)
        assert logits.shape == (2, 6, 100)
        assert nh.shape == (2, 6, 32)

    def test_draft_sequence_length(self):
        torch.manual_seed(0)
        head = EagleDraftHead(d_model=32, num_heads=4, vocab_size=100)
        hidden = torch.randn(1, 1, 32)
        draft = head.draft_sequence(hidden, 3)
        assert len(draft) == 3
        assert all(0 <= d < 100 for d in draft)

    def test_draft_sequence_zero(self):
        torch.manual_seed(0)
        head = EagleDraftHead(d_model=32, num_heads=4, vocab_size=100)
        assert head.draft_sequence(torch.randn(1, 1, 32), 0) == []


class TestTrainEagle:
    def test_training_reduces_loss(self):
        _tok, model, _engine, _type_of, head, pairs = _make_env()
        loss_before = _eval_loss(head, pairs)
        train_eagle(head, model, pairs, steps=3, lr=5e-3)
        loss_after = _eval_loss(head, pairs)
        assert loss_after < loss_before

    def test_collect_hidden_pairs_shape(self):
        _, _model, _, _type_of, _, pairs = _make_env()
        assert len(pairs) >= 1


def _eval_loss(head, pairs):
    total = torch.tensor(0.0)
    n = 0
    for hidden, target in pairs:
        logits = head.predict_next(hidden)[0]
        total = total + torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
        )
        n += 1
    return float(total.detach() / max(n, 1))


class TestSpeculativeEagle:
    def test_greedy_preserving(self):
        _tok, model, engine, _type_of, head, pairs = _make_env()
        train_eagle(head, model, pairs, steps=5, lr=1e-3)
        greedy = engine.greedy("create a steel box", max_len=10, use_cache=True)
        eagle = speculative_eagle(
            engine, "create a steel box", head, max_len=10, num_speculative_tokens=3
        )
        assert eagle.ids == greedy.ids

    def test_respects_max_len(self):
        _tok, model, engine, _type_of, head, pairs = _make_env()
        train_eagle(head, model, pairs, steps=5, lr=1e-3)
        result = speculative_eagle(
            engine, "create a steel box", head, max_len=10, num_speculative_tokens=2
        )
        assert len(result.ids) <= 10  # max_len content tokens
