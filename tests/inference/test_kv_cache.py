"""
KV-cache incremental decoding tests (P1).

Verifies that ``GeometryAwareTransformer.prepare_decoder_cache`` +
``decode_step`` produce exactly the same logits as the full-sequence
``forward`` when the decoder is composed of per-token subsystems, and that
the engine's cached greedy path matches the uncached path token-for-token.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _make_pure_config() -> CADConfig:
    """Decoder made only of per-token subsystems: self + geometry attention."""
    cfg = CADConfig.mini()
    cfg.model.geometry_attn_heads = 2
    cfg.model.constraint_attn_heads = 0
    cfg.model.memory_attn_heads = 0
    cfg.model.agent_attn_heads = 0
    cfg.model.uncertainty_attn_heads = 0
    cfg.model.use_moe = False
    cfg.model.num_encoder_layers = 2
    cfg.model.num_decoder_layers = 2
    return cfg


def _make_default_config() -> CADConfig:
    cfg = CADConfig.mini()
    cfg.model.use_moe = False
    cfg.model.num_encoder_layers = 2
    cfg.model.num_decoder_layers = 2
    return cfg


def _make_tok() -> AutonomousCADTokenizer:
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box", "make a hole"])
    return tok


def _logits_all_steps(model, cache, tgt_ids, tgt_type_ids):
    """Replay a full sequence through decode_step; returns per-step logits."""
    step_logits = []
    for pos in range(tgt_ids.shape[1]):
        logits, _ = model.decode_step(
            tgt_ids[:, pos : pos + 1], tgt_type_ids[:, pos : pos + 1], cache
        )
        step_logits.append(logits)
    return step_logits


@pytest.mark.parametrize("backend", ["math", "sdpa", "gqa", "mla", "linear"])
def test_decode_step_matches_full_forward(backend: str):
    torch.manual_seed(0)
    cfg = _make_pure_config()
    cfg.model.attention_backend = backend
    model = GeometryAwareTransformer(cfg)
    model.eval()

    tok = _make_tok()
    prompt = tok.encode_text("create a steel box")
    src = torch.tensor([prompt])
    cad_ids = [tok.vocab["SKETCH_RECT"], tok.vocab["EXTRUDE"], tok.vocab["BOX"], tok.eos_id]
    tgt_in = torch.tensor([cad_ids])
    tgt_type = torch.tensor([[tok.vocab.type_id_of(int(i)) for i in cad_ids]])

    with torch.no_grad():
        full_logits, _ = model(src, tgt_in, tgt_type, src_key_padding_mask=(src == tok.pad_id))

        cache = model.prepare_decoder_cache(src, src_key_padding_mask=(src == tok.pad_id))
        step_logits = _logits_all_steps(model, cache, tgt_in, tgt_type)

    assert len(step_logits) == tgt_in.shape[1]
    for pos, step in enumerate(step_logits):
        assert step.shape == (1, 1, model.cad_vocab_size)
        diff = (step[0, 0] - full_logits[0, pos]).abs().max().item()
        assert diff < 1e-4, f"backend={backend} pos={pos} max diff {diff}"


def test_cached_greedy_matches_uncached():
    torch.manual_seed(0)
    cfg = _make_pure_config()
    model = GeometryAwareTransformer(cfg)
    tok = _make_tok()
    engine = CADInferenceEngine(model, tok, device="cpu")

    r1 = engine.greedy("create a steel box", max_len=16, use_cache=False)
    r2 = engine.greedy("create a steel box", max_len=16, use_cache=True)

    assert r1.ids == r2.ids, f"cached {r1.ids} != uncached {r2.ids}"
    assert r1.stopped_on_eos == r2.stopped_on_eos
    assert len(r1.per_token_confidence) == len(r2.per_token_confidence)
    # Confidence values must agree closely too (same head activations).
    for a, b in zip(r1.per_token_confidence, r2.per_token_confidence, strict=True):
        assert abs(a - b) < 1e-3


def test_cached_greedy_default_config_runs():
    torch.manual_seed(0)
    cfg = _make_default_config()  # agent + uncertainty heads active
    model = GeometryAwareTransformer(cfg)
    tok = _make_tok()
    engine = CADInferenceEngine(model, tok, device="cpu")

    res = engine.greedy("make a hole", max_len=16, use_cache=True)
    assert res.ids, "expected a non-empty generated sequence"
    assert 0 < len(res.per_token_confidence) == len(res.ids)
    assert res.confidence > 0.0

    # Deterministic: same model, same seed, same cache path → same output.
    res2 = engine.greedy("make a hole", max_len=16, use_cache=True)
    assert res2.ids == res.ids


def test_prepare_cache_invariants():
    torch.manual_seed(0)
    model = GeometryAwareTransformer(_make_pure_config())
    model.eval()
    tok = _make_tok()
    src = torch.tensor([tok.encode_text("create a steel box")])

    with torch.no_grad():
        cache = model.prepare_decoder_cache(src, src_key_padding_mask=(src == tok.pad_id))

    assert cache["encoder_hidden_states"].shape[1] == len(src[0])
    assert len(cache["geometry_kv"]) == model.config.model.num_decoder_layers
    assert len(cache["memory_kv"]) == model.config.model.num_decoder_layers
    assert len(cache["blocks"]) == model.config.model.num_decoder_layers
    assert cache["position_offset"] == 0
    # Geometry K/V precomputed for every block.
    for i, kv in enumerate(cache["geometry_kv"]):
        assert kv is not None, f"block {i} missing geometry kv"
        k, _ = kv
        assert k.shape[2] == len(src[0])  # one key per source token


# ----------------------------------------------------------- speculative


def test_ngram_draft():
    engine = CADInferenceEngine(None, _make_tok(), device="cpu")
    # (1, 2) recurs at index 0; the following tokens are [3, 4, 5].
    ids = [1, 2, 3, 4, 5, 1, 2]
    assert engine._ngram_draft(ids, n=2, k=3) == [3, 4, 5]
    assert engine._ngram_draft([1, 2, 3, 4], n=2, k=2) == []  # no repetition
    assert engine._ngram_draft([1, 2, 3], n=4, k=2) == []  # n-gram longer than history


def test_speculative_matches_greedy():
    torch.manual_seed(0)
    model = GeometryAwareTransformer(_make_pure_config())
    tok = _make_tok()
    engine = CADInferenceEngine(model, tok, device="cpu")

    # Speculative decoding is greedy-*preserving*: it must reproduce the
    # target model's argmax exactly.  Compare against the *uncached* greedy
    # reference (both use full-sequence forwards → bit-identical logits).
    # Comparing against cached greedy would also entangle the separate
    # "cached ≡ full" guarantee, whose ~2e-7 float noise can flip a near-tied
    # argmax on an untrained model (see test_decode_step_matches_full_forward).
    greedy = engine.greedy("create a steel box", max_len=12, use_cache=False)
    spec = engine.speculative("create a steel box", max_len=12, num_speculative_tokens=4)

    assert spec.ids == greedy.ids, f"spec {spec.ids} != greedy {greedy.ids}"
    assert spec.stopped_on_eos == greedy.stopped_on_eos


def test_speculative_rejects_wrong_drafts():
    """The engine must never commit a draft token the target model would not
    greedily produce, even when the drafter proposes nonsense."""
    torch.manual_seed(0)
    model = GeometryAwareTransformer(_make_pure_config())
    tok = _make_tok()
    engine = CADInferenceEngine(model, tok, device="cpu")

    greedy = engine.greedy("create a steel box", max_len=10, use_cache=True)

    # Monkeypatch the drafter to always propose a fixed (wrong) token.
    wrong = tok.vocab["EXTRUDE"]

    def bad_draft(ids, n, k):
        return [wrong] * k

    engine._ngram_draft = bad_draft
    spec = engine.speculative("create a steel box", max_len=10, num_speculative_tokens=3)

    assert spec.ids == greedy.ids, f"spec {spec.ids} != greedy {greedy.ids}"
    assert tok.eos_id not in spec.ids or spec.ids[-1] == tok.eos_id
