"""
tests/transformer/test_ssm.py
=============================
Tests for the hybrid Gated DeltaNet SSM layer (linear-time recurrence).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.ssm import GatedDeltaNet, add_ssm_blocks


class TestGatedDeltaNet:
    def test_shapes_and_divisibility(self):
        net = GatedDeltaNet(d_model=64, heads=4)
        x = torch.randn(2, 8, 64)
        out = net(x)
        assert out.shape == (2, 8, 64)
        assert torch.isfinite(out).all()

    def test_divisibility_validation(self):
        with pytest.raises(ValueError, match="divisible"):
            GatedDeltaNet(d_model=63, heads=4)

    def test_state_shape(self):
        net = GatedDeltaNet(d_model=64, heads=4)
        state = net._initial_state(2, torch.device("cpu"))
        assert state.shape == (2, 4, 16)

    def test_cached_matches_full_recurrence(self):
        torch.manual_seed(0)
        net = GatedDeltaNet(d_model=32, heads=4)
        net.eval()
        B, T = 2, 6
        x = torch.randn(B, T, 32)
        with torch.no_grad():
            full = net(x)
            state = net._initial_state(B, x.device)
            outs = []
            for t in range(T):
                step, state = net.forward_cached(x[:, t : t + 1], state)
                outs.append(step)
            cached = torch.cat(outs, dim=1)
        diff = (full - cached).abs().max().item()
        assert diff < 1e-5

    def test_cached_recurrence_participates_in_autograd(self):
        """forward_cached must not be @torch.no_grad: gradients must flow
        through the recurrence (v6.1 §4.3)."""
        torch.manual_seed(0)
        net = GatedDeltaNet(d_model=32, heads=4)
        net.train()
        net.dropout.p = 0.0
        x = torch.randn(1, 4, 32)
        state = net._initial_state(1, x.device)
        outs = []
        for t in range(4):
            step, state = net.forward_cached(x[:, t : t + 1], state)
            outs.append(step)
        out = torch.cat(outs, dim=1)
        assert out.requires_grad, "forward_cached output detached from autograd"
        loss = out.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, "no gradients produced"
        for g in grads:
            assert torch.isfinite(g).all()

    def test_cached_matches_full_in_train_mode_without_dropout(self):
        """With dropout disabled the cached replay equals the full
        recurrence in train mode (the correct equivalence contract)."""
        torch.manual_seed(0)
        net = GatedDeltaNet(d_model=32, heads=4)
        net.train()
        net.dropout.p = 0.0
        B, T = 2, 6
        x = torch.randn(B, T, 32)
        full = net(x)
        state = net._initial_state(B, x.device)
        outs = []
        for t in range(T):
            step, state = net.forward_cached(x[:, t : t + 1], state)
            outs.append(step)
        cached = torch.cat(outs, dim=1)
        diff = (full - cached).abs().max().item()
        assert diff < 1e-5

    def test_recurrence_is_linear_in_memory(self):
        # forward uses a fixed-size recurrent state (no KV growth).
        net = GatedDeltaNet(d_model=32, heads=4)
        state = net._initial_state(1, torch.device("cpu"))
        assert state.numel() == 4 * 8


class TestAddSsmBlocks:
    def test_interleave_plan(self):
        blocks = add_ssm_blocks(6, 32, every_n=3, heads=4)
        assert len(blocks) == 6
        assert [b is not None for b in blocks] == [False, False, True, False, False, True]
        assert all(isinstance(b, GatedDeltaNet) for b in blocks if b is not None)

    def test_invalid_every_n(self):
        with pytest.raises(ValueError, match="every_n"):
            add_ssm_blocks(4, 32, every_n=0)
