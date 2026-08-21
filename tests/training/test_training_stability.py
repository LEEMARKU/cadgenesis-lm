"""
tests/training/test_training_stability.py
=========================================
Training-stability regression tests (v6.1 §4.1).

Regression history
------------------
v6.0: every packed-sequence training run produced ``train=nan val=nan``.
Root cause: ``encode``/``decode`` merged the padding mask into the packed
block-diagonal masks by *addition*, re-killing already-repaired dead rows
(all-(-inf) score rows) → softmax(0/0) → NaN pollution.  Fixed by
``safe_softmax`` + ``repair_fully_masked_rows``.

These tests pin that behaviour: a packed mini-model training step must
always produce finite loss, finite gradients, and a loss that actually
decreases across a few optimizer steps.
"""

from __future__ import annotations

import json

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.datasets.cad_jsonl import CADJsonlDataset
from cadgenesis.datasets.cad_program_synth import build_synthetic_records, token_coverage
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.vocabulary import TokenFamily
from cadgenesis.training.packing import pack_batch
from cadgenesis.training.trainer import CADTrainer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _build_tokenizer(tmp_path) -> AutonomousCADTokenizer:
    torch.manual_seed(0)
    records = build_synthetic_records(24, seed=0)
    jsonl = tmp_path / "progs.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    tok = AutonomousCADTokenizer.build_mini()
    tok2id = tok.vocab.to_tok2id()
    missing = sorted(t for t in token_coverage(records) if tok2id.get(t) is None)
    if missing:
        tok.vocab.register_many(
            [
                (
                    t,
                    TokenFamily.NUMERIC if t.startswith("NUM_") else TokenFamily.FEATURE,
                )
                for t in missing
            ]
        )
    tok.build_lang_vocab([r.get("text", "") for r in records])
    return tok


def _packed_batch(tok: AutonomousCADTokenizer, tmp_path) -> dict:
    cfg = CADConfig.mini()
    base = CADJsonlDataset(str(tmp_path / "progs.jsonl"), tok)

    def collate(batch):
        return pack_batch(
            batch,
            max_src_len=cfg.training.packed_max_src_len,
            max_tgt_len=cfg.training.packed_max_tgt_len,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            pad_id=tok.pad_id,
            seed=0,
        )

    from torch.utils.data import DataLoader

    dl = DataLoader(
        base,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(0),
    )
    return next(iter(dl))


def test_packed_training_step_loss_and_grads_are_finite(tmp_path):
    """One packed forward+backward must yield finite loss and gradients."""
    torch.manual_seed(0)
    tok = _build_tokenizer(tmp_path)
    cfg = CADConfig.mini()
    model = GeometryAwareTransformer(cfg)
    trainer = CADTrainer(config=cfg, model=model, tokenizer=tok, device="cpu")
    batch = _packed_batch(tok, tmp_path)

    total, breakdown = trainer._packed_loss(batch)
    assert torch.isfinite(total), f"packed loss is not finite: {total.item()}"
    assert total.item() > 0.0

    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    non_finite = [name for name, p in model.named_parameters()
                  if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not non_finite, f"non-finite gradients: {non_finite}"


def test_packed_training_loss_decreases_across_steps(tmp_path):
    """A few optimizer steps on the packed path must reduce the loss."""
    torch.manual_seed(0)
    tok = _build_tokenizer(tmp_path)
    cfg = CADConfig.mini()
    model = GeometryAwareTransformer(cfg)
    trainer = CADTrainer(config=cfg, model=model, tokenizer=tok, device="cpu")
    batch = _packed_batch(tok, tmp_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        total, _ = trainer._packed_loss(batch)
        assert torch.isfinite(total)
        total.backward()
        optimizer.step()
        losses.append(total.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_gradient_checkpointing_matches_plain_forward(tmp_path):
    """Activation checkpointing must be numerically identical to the plain
    forward (v6.1 §4.2) and must activate only in training mode."""
    torch.manual_seed(0)
    tok = _build_tokenizer(tmp_path)
    cfg = CADConfig.mini()
    cfg.training.gradient_checkpointing = True
    # Dropout is disabled so the comparison is exact: checkpointing forks
    # the RNG stream (preserving it for the backward replay), so with active
    # dropout the two runs legitimately differ by dropout noise.
    cfg.model.dropout = 0.0
    model = GeometryAwareTransformer(cfg)
    trainer = CADTrainer(config=cfg, model=model, tokenizer=tok, device="cpu")
    batch = _packed_batch(tok, tmp_path)

    assert model.training, "test precondition: model in train mode"

    loss_ckpt, _ = trainer._packed_loss(batch)
    loss_ckpt.backward()
    grads_ckpt = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert grads_ckpt, "no gradients under checkpointing"
    assert all(torch.isfinite(g).all() for g in grads_ckpt.values())

    model.zero_grad(set_to_none=True)
    cfg.training.gradient_checkpointing = False
    loss_plain, _ = trainer._packed_loss(batch)
    loss_plain.backward()
    assert torch.allclose(loss_ckpt, loss_plain, atol=1e-5), (
        f"checkpointed loss {loss_ckpt.item()} != plain loss {loss_plain.item()}"
    )
    for name, p in model.named_parameters():
        if p.grad is None:
            assert name not in grads_ckpt
        else:
            assert name in grads_ckpt
            assert torch.allclose(
                p.grad, grads_ckpt[name], atol=1e-5
            ), f"grad mismatch for {name} under checkpointing"

    # Evaluation mode must bypass checkpointing (no behavioural change).
    cfg.training.gradient_checkpointing = True
    model.eval()
    with torch.no_grad():
        out, _ = model(batch["src"], batch["tgt"][:, :-1], torch.zeros_like(batch["tgt"][:, :-1]))
    assert torch.isfinite(out).all()