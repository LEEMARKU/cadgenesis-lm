"""tests/transformer/test_stacks.py
================================
Unit tests for cadgenesis.transformer.encoder and .decoder.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.decoder import DecoderStack
from cadgenesis.transformer.encoder import EncoderStack


@pytest.fixture
def mini_config() -> CADConfig:
    return CADConfig.mini()


class TestEncoderStack:
    def test_shape(self, mini_config):
        enc = EncoderStack(mini_config.model)
        x = torch.randn(2, 16, mini_config.model.d_model)
        out = enc(x)
        assert out.shape == (2, 16, mini_config.model.d_model)

    def test_num_layers(self, mini_config):
        enc = EncoderStack(mini_config.model)
        assert enc.num_layers == mini_config.model.num_encoder_layers

    def test_layer_gate_hook(self, mini_config):
        enc = EncoderStack(mini_config.model)
        x = torch.randn(2, 16, mini_config.model.d_model)
        calls = {"n": 0}

        def gate(idx, h, stage):
            calls["n"] += 1
            return None

        out = enc(x, layer_gate=gate)
        assert out.shape == (2, 16, mini_config.model.d_model)
        assert calls["n"] == mini_config.model.num_encoder_layers

    def test_refine_fn(self, mini_config):
        enc = EncoderStack(mini_config.model)
        x = torch.randn(2, 16, mini_config.model.d_model)
        mem = torch.randn(2, 8, mini_config.model.d_model)
        calls = {"n": 0}

        def refine(m, h):
            calls["n"] += 1
            return m

        enc(x, memory_bank=mem, refine_fn=refine)
        assert calls["n"] == mini_config.model.num_encoder_layers

    def test_backward(self, mini_config):
        enc = EncoderStack(mini_config.model)
        x = torch.randn(2, 16, mini_config.model.d_model)
        enc(x).sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)

    def test_validation(self):
        cfg = CADConfig.mini().model
        cfg.num_encoder_layers = 0
        with pytest.raises(ValueError):
            EncoderStack(cfg)


class TestDecoderStack:
    def test_shape(self, mini_config):
        dec = DecoderStack(mini_config.model)
        x = torch.randn(2, 8, mini_config.model.d_model)
        enc_states = torch.randn(2, 16, mini_config.model.d_model)
        out, conf = dec(x, encoder_hidden_states=enc_states)
        assert out.shape == (2, 8, mini_config.model.d_model)
        assert conf is not None and conf.shape[-1] == 1

    def test_agent_fn(self, mini_config):
        dec = DecoderStack(mini_config.model)
        x = torch.randn(2, 8, mini_config.model.d_model)
        mem = torch.randn(2, 8, mini_config.model.d_model)
        calls = {"n": 0}

        def agent_fn(h, m):
            calls["n"] += 1
            return h

        _, _ = dec(x, memory_bank=mem, agent_fn=agent_fn)
        assert calls["n"] == mini_config.model.num_decoder_layers

    def test_causal_is_applied(self, mini_config):
        dec = DecoderStack(mini_config.model)
        x = torch.randn(2, 4, mini_config.model.d_model)
        out, _ = dec(x)
        assert torch.isfinite(out).all()

    def test_backward(self, mini_config):
        dec = DecoderStack(mini_config.model)
        x = torch.randn(2, 8, mini_config.model.d_model)
        enc = torch.randn(2, 16, mini_config.model.d_model)
        out, conf = dec(x, encoder_hidden_states=enc)
        (out.sum() + conf.sum()).backward()
        assert all(p.grad is not None for p in dec.parameters() if p.requires_grad)

    def test_validation(self):
        cfg = CADConfig.mini().model
        cfg.num_decoder_layers = 0
        with pytest.raises(ValueError):
            DecoderStack(cfg)
