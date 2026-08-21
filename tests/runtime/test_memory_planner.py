"""
tests/runtime/test_memory_planner.py
====================================
Memory planning (v6.2): estimates are conservative and recommendations fit
their preset.
"""

from __future__ import annotations

from cadgenesis.runtime.hardware import PRESETS
from cadgenesis.runtime.memory_planner import (
    estimate_training_memory,
    fits,
    recommend_config_overrides,
)


def test_estimate_monotonic_in_batch_and_seq():
    base = dict(n_params=2_000_000, d_model=128, num_layers=3, vocab_size=1024)
    small = estimate_training_memory(seq_len=64, batch_size=1, **base)
    bigger_batch = estimate_training_memory(seq_len=64, batch_size=2, **base)
    bigger_seq = estimate_training_memory(seq_len=128, batch_size=1, **base)
    assert bigger_batch.total_bytes > small.total_bytes
    assert bigger_seq.total_bytes > small.total_bytes


def test_estimate_includes_optimizer_and_gradients():
    est = estimate_training_memory(
        n_params=1_000_000, d_model=128, num_layers=1, seq_len=16, batch_size=1, vocab_size=256
    )
    # AdamW: 2 fp32 copies; params+grads: 2x2 bytes each
    assert est.params_bytes == 2_000_000
    assert est.gradients_bytes == 2_000_000
    assert est.optimizer_bytes == 8_000_000
    assert est.total_bytes > est.params_bytes + est.gradients_bytes + est.optimizer_bytes


def test_checkpointing_reduces_activations():
    kw = dict(
        n_params=1_000_000,
        d_model=128,
        num_layers=8,
        seq_len=256,
        batch_size=4,
        vocab_size=1024,
    )
    plain = estimate_training_memory(**kw, grad_checkpointing=False)
    ckpt = estimate_training_memory(**kw, grad_checkpointing=True)
    assert ckpt.activations_bytes < plain.activations_bytes


def test_mini_model_fits_gtx1650():
    p = PRESETS["gtx1650_4gb"]
    est = estimate_training_memory(
        n_params=2_597_660, d_model=128, num_layers=3, seq_len=64, batch_size=8, vocab_size=1024
    )
    assert fits(p, est)


def test_recommendation_fits_when_already_small():
    p = PRESETS["gtx1650_4gb"]
    rec = recommend_config_overrides(
        p,
        n_params=2_597_660,
        d_model=128,
        num_layers=3,
        vocab_size=1024,
        train_batch=8,
        max_seq_len=64,
        grad_checkpointing=True,
    )
    assert rec.fits_without_changes
    assert rec.max_train_batch == 8
    assert rec.max_seq_len == 64


def test_recommendation_shrinks_to_fit():
    p = PRESETS["gtx1650_4gb"]
    rec = recommend_config_overrides(
        p,
        n_params=48_000_000,
        d_model=384,
        num_layers=6,
        vocab_size=3392,
        train_batch=256,
        max_seq_len=2048,
        grad_checkpointing=False,
    )
    assert not rec.fits_without_changes
    assert rec.max_train_batch <= 256
    assert rec.max_seq_len <= 2048
    assert fits(p, rec.estimate)


def test_recommendation_never_below_one():
    p = PRESETS["gtx1650_4gb"]
    rec = recommend_config_overrides(
        p,
        n_params=500_000_000,
        d_model=1024,
        num_layers=24,
        vocab_size=3392,
        train_batch=1,
        max_seq_len=1,
        grad_checkpointing=False,
    )
    assert rec.max_train_batch >= 1
    assert rec.max_seq_len >= 1
    assert rec.enable_grad_checkpointing is True


def test_cpu_fits_uses_system_ram():
    p = PRESETS["cpu"]
    est = estimate_training_memory(
        n_params=2_597_660, d_model=128, num_layers=3, seq_len=64, batch_size=8, vocab_size=1024
    )
    assert fits(p, est)