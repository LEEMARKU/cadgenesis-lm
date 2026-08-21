"""
tests/inference/test_beam.py
============================
Beam search semantics (v6.1 §4.8):

* EOS handling — finished beams are retired and never re-expanded;
* length normalization — GNMT-style penalty ranks hypotheses by
  ``score / ((5 + len) / 6) ** alpha``;
* score normalization — the final pick is the best *normalized* hypothesis
  across finished and unfinished beams.

A scripted stub model supplies exact log-probabilities so the search
behaviour is fully deterministic.
"""

from __future__ import annotations

import math

import pytest
import torch

from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.tokenizer import AutonomousCADTokenizer

X, Y = 10, 11


class ScriptedModel(torch.nn.Module):
    """Logits follow a hand-written script:

    * step 1: P(X) = 0.41, P(Y) = 0.61  (cumulative -0.891 / -0.495)
    * last == X: token X until T == 9, then EOS with P = 1
      → cumulative ≈ -0.891, length 10 (incl. BOS)
    * last == Y: EOS at T == 2 with P = 1  → cumulative -0.495, length 3

    Raw scores prefer Y (-0.495 > -0.891); the GNMT-normalized scores
    (alpha=1.0) prefer X (-0.891/2.5 > -0.495/1.333).
    """

    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, src, tgt, tgt_type, src_key_padding_mask=None):
        B, T = tgt.shape
        logits = torch.full((B, T, self.vocab_size), float("-inf"))
        last = tgt[0, -1].item()
        if T == 1:
            logits[0, -1, X] = math.log(0.41)
            logits[0, -1, Y] = math.log(0.61)
        elif last == X:
            logits[0, -1, 2 if T == 9 else X] = 0.0
        elif last == Y:
            logits[0, -1, 2 if T == 2 else Y] = 0.0
        else:
            raise AssertionError(f"unexpected target tail: id={last} len={T}")
        return logits, None


@pytest.fixture
def engine():
    tokenizer = AutonomousCADTokenizer.build_mini()
    model = ScriptedModel()
    return CADInferenceEngine(model, tokenizer, device="cpu")


@pytest.fixture
def engine_eos():
    """EOS-termination script: emit token A until len == 4, then EOS (P=1)."""
    tokenizer = AutonomousCADTokenizer.build_mini()

    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 100

        def forward(self, src, tgt, tgt_type, src_key_padding_mask=None):
            B, T = tgt.shape
            logits = torch.full((B, T, self.vocab_size), float("-inf"))
            logits[0, -1, 2 if T == 4 else X] = 0.0
            return logits, None

    return CADInferenceEngine(EosModel(), tokenizer, device="cpu")


def test_beam_width_must_be_positive(engine):
    with pytest.raises(ValueError, match="beam_width"):
        engine.beam("make a box", beam_width=0)


def test_eos_terminates_and_is_never_extended(engine_eos):
    res = engine_eos.beam("make a box", beam_width=3, max_len=12)
    assert res.ids == [X, X, X, 2], f"unexpected ids: {res.ids}"
    assert res.stopped_on_eos


def test_raw_scores_prefer_longer_path(engine):
    res = engine.beam("make a box", beam_width=3, max_len=12, length_penalty=0.0)
    # Y wins on plain cumulative log-probability (-0.495 > -0.891).
    assert res.ids[0] == Y
    assert res.ids[-1] == 2


def test_length_normalization_prefers_shorter_path(engine):
    res = engine.beam("make a box", beam_width=3, max_len=12, length_penalty=1.0)
    # X wins normalized: -0.891/7 > -0.495/3.
    assert res.ids[0] == X
    assert res.ids[-1] == 2


def test_finished_and_unfinished_compared_by_normalized_score():
    """max_len truncation: an unfinished beam with a good normalized score
    beats a finished hypothesis with a worse one (and vice versa raw)."""
    tokenizer = AutonomousCADTokenizer.build_mini()

    class TruncatingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 100

        def forward(self, src, tgt, tgt_type, src_key_padding_mask=None):
            B, T = tgt.shape
            logits = torch.full((B, T, self.vocab_size), float("-inf"))
            last = tgt[0, -1].item()
            if T == 1:
                # Half the mass on the instant-EOS path, half on the long path.
                logits[0, -1, 2] = math.log(0.5)
                logits[0, -1, X] = math.log(0.5)
            elif last == X:
                logits[0, -1, X] = math.log(0.99)  # never EOS within max_len
            return logits, None

    engine = CADInferenceEngine(TruncatingModel(), tokenizer, device="cpu")

    # Raw scores: the EOS hypothesis (-0.693) beats the truncated X path
    # (-0.693 + log 0.99 ≈ -0.703), so it is selected.
    raw = engine.beam("make a box", beam_width=2, max_len=2, length_penalty=0.0)
    assert raw.ids == [2] and raw.stopped_on_eos

    # Normalized (alpha=1): -0.703/3 > -0.693/2, so the *unfinished* beam wins.
    norm = engine.beam("make a box", beam_width=2, max_len=2, length_penalty=1.0)
    assert norm.ids == [X, X] and not norm.stopped_on_eos