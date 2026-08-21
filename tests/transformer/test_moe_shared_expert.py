"""
tests/transformer/test_moe_shared_expert.py
===========================================
Tests for the DeepSeek-V3-style shared expert fused into the sparse MoE.
"""

from __future__ import annotations

import torch

from cadgenesis.transformer.moe import SparseMoEFFN


class TestSharedExpert:
    def test_no_shared_expert_by_default(self):
        moe = SparseMoEFFN(d_model=32, num_experts=4, top_k=2)
        assert moe.shared_expert is None
        assert moe.num_shared_experts == 0

    def test_shared_expert_always_active(self):
        torch.manual_seed(0)
        moe = SparseMoEFFN(
            d_model=32,
            num_experts=4,
            top_k=2,
            num_shared_experts=2,
            shared_expert_dim=16,
        )
        assert moe.shared_expert is not None
        x = torch.randn(2, 5, 32)
        out = moe(x)
        assert out.shape == (2, 5, 32)
        assert torch.isfinite(out).all()

    def test_shared_expert_adds_contribution(self):
        torch.manual_seed(0)
        shared_dim = 16
        moe_on = SparseMoEFFN(
            d_model=32,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            shared_expert_dim=shared_dim,
        )
        moe_off = SparseMoEFFN(d_model=32, num_experts=4, top_k=2)
        # Copy routed-expert weights so the only difference is the shared expert.
        moe_off.router.load_state_dict(moe_on.router.state_dict())
        moe_off.expert_bias.data.copy_(moe_on.expert_bias.data)
        for a, b in zip(moe_on.experts, moe_off.experts, strict=False):
            b.load_state_dict(a.state_dict())
        # Re-init the shared expert away from zero so its output is non-trivial.
        with torch.no_grad():
            for p in moe_on.shared_expert.parameters():
                p.uniform_(-0.5, 0.5)

        x = torch.randn(2, 5, 32)
        moe_on.eval()
        moe_off.eval()
        with torch.no_grad():
            y_on = moe_on(x)
            y_off = moe_off(x)
            y_shared = moe_on.shared_expert(x.reshape(-1, 32)).view(2, 5, 32)
        assert not torch.allclose(y_on, y_off, atol=1e-6)
        assert torch.allclose(y_on - y_off, y_shared, atol=1e-5)

    def test_negative_count_rejected(self):
        try:
            SparseMoEFFN(d_model=32, num_experts=4, top_k=2, num_shared_experts=-1)
        except ValueError as exc:
            assert "num_shared_experts" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_aux_loss_still_reported(self):
        torch.manual_seed(0)
        moe = SparseMoEFFN(
            d_model=32,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            shared_expert_dim=16,
        )
        moe(torch.randn(2, 5, 32))
        assert torch.isfinite(moe.get_aux_loss())
