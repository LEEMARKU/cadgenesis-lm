"""
tests/training/test_mu_transfer.py
==================================
Tests for µTransfer (maximal update parametrization): init scaling and
LR-multiplier param groups.
"""

from __future__ import annotations

import torch

from cadgenesis.config import CADConfig
from cadgenesis.training.mu_transfer import (
    apply_mu_transfer,
    build_mu_optimizer,
    mu_param_groups,
)
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


class TestMuTransfer:
    def test_readout_initialised_to_1_over_d(self):
        torch.manual_seed(0)
        cfg = CADConfig()
        d = cfg.model.d_model
        model = GeometryAwareTransformer(cfg)
        before = model.out_proj.weight.clone()
        apply_mu_transfer(model, d_model=d)
        after = model.out_proj.weight
        # Readout scaled by sqrt(d): std 1/sqrt(d) -> 1/d.
        ratio = (after / before).float().abs().mean()
        assert ratio == torch.tensor(d**-0.5)

    def test_embedding_output_scaled_forward(self):
        torch.manual_seed(0)
        cfg = CADConfig()
        d = cfg.model.d_model
        model = GeometryAwareTransformer(cfg)
        apply_mu_transfer(model, d_model=d)
        # The CAD input embedding (also the tied LM head) is wrapped with the
        # 1/sqrt(d) output scale.  Rows start at std 0.02, so the scaled RMS
        # must be ~ 0.02 (unscaled it would be ~ 0.02 * sqrt(d)).
        x = torch.tensor([[5, 6, 7]])
        out = model.cad_embed(x)
        assert out.shape == (1, 3, d)
        rms = out.float().square().mean().sqrt()
        assert rms.item() < 0.1
        # The wrapper is actually in place: output == stock embed * 1/sqrt(d).
        raw = torch.nn.Embedding(model.cad_vocab_size, d)
        raw.weight.data.copy_(model.cad_embed.weight.data)
        assert torch.allclose(model.cad_embed(x), raw(x) * (1.0 / d**0.5), atol=1e-6)

    def test_param_groups_lr_multipliers(self):
        torch.manual_seed(0)
        cfg = CADConfig()
        d = cfg.model.d_model
        model = GeometryAwareTransformer(cfg)
        groups = mu_param_groups(model, base_lr=1e-3, d_model=d)
        lrs = [g["lr"] for g in groups]
        assert lrs == [1e-3 * d, 1e-3, 1e-3]
        # Readout group holds the LM head (tied to cad_embed -> same tensor).
        readout_params = set(groups[0]["params"])
        assert model.out_proj.weight in readout_params
        # Bias / norm group has no weight decay.
        assert groups[2]["weight_decay"] == 0.0
        assert groups[0]["weight_decay"] == 0.1
        # Every parameter belongs to exactly one group.
        grouped = sum(len(g["params"]) for g in groups)
        total = sum(1 for _ in model.parameters())
        assert grouped == total

    def test_build_mu_optimizer(self):
        torch.manual_seed(0)
        cfg = CADConfig()
        model = GeometryAwareTransformer(cfg)
        opt = build_mu_optimizer(model, base_lr=1e-3, d_model=cfg.model.d_model)
        assert isinstance(opt, torch.optim.AdamW)
        assert len(opt.param_groups) == 3

    def test_apply_mu_is_inplace(self):
        torch.manual_seed(0)
        cfg = CADConfig()
        model = GeometryAwareTransformer(cfg)
        result = apply_mu_transfer(model, d_model=cfg.model.d_model)
        assert result is model

    def test_trains_after_mu_transfer(self):
        torch.manual_seed(0)
        cfg = CADConfig()
        model = GeometryAwareTransformer(cfg)
        apply_mu_transfer(model, d_model=cfg.model.d_model)
        opt = build_mu_optimizer(model, base_lr=1e-3, d_model=cfg.model.d_model)
        src = torch.randint(0, 50, (2, 8))
        tgt = torch.randint(0, 30, (2, 4))
        tt = torch.randint(0, 3, (2, 4))
        logits, _ = model(src, tgt, tt)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.cad_vocab_size), tgt.reshape(-1)
        )
        loss.backward()
        opt.step()
        opt.zero_grad()
        assert torch.isfinite(loss)
