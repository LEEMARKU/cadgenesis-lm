"""tests/transformer/test_specialized_moe.py
============================================
Unit tests for the specialized (domain) MoE module (Pillar 1).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.transformer.specialized_moe import (
    DEFAULT_DOMAIN_EXPERTS,
    DomainExpert,
    SpecializedMoEFFN,
    register_expert_type,
    registered_expert_types,
)


class TestDomainExpert:
    def test_forward(self):
        expert = DomainExpert("geometry", d_model=32, expert_dim=64)
        x = torch.randn(4, 32)
        out = expert(x)
        assert out.shape == (4, 32)
        out.sum().backward()
        assert expert.w1.weight.grad is not None


class TestSpecializedMoE:
    def test_shape_and_grad(self):
        moe = SpecializedMoEFFN(d_model=64, experts_per_domain=2, top_k=2)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == (2, 8, 64)
        out.sum().backward()
        assert moe.router.weight.grad is not None

    def test_default_domains(self):
        moe = SpecializedMoEFFN(d_model=32, experts_per_domain=1, top_k=2)
        assert set(moe.expert_types) == set(DEFAULT_DOMAIN_EXPERTS)
        assert moe.num_experts == len(DEFAULT_DOMAIN_EXPERTS)
        assert moe.domain_of(0) == "geometry"

    def test_expert_domains_labels(self):
        moe = SpecializedMoEFFN(d_model=32, experts_per_domain=2, top_k=2)
        labels = moe.expert_domains()
        assert len(labels) == 10
        assert labels[:2] == ["geometry", "geometry"]
        assert labels[2:4] == ["manufacturing", "manufacturing"]

    def test_load_balance_loss_positive(self):
        moe = SpecializedMoEFFN(d_model=32, experts_per_domain=2, top_k=2)
        moe(torch.randn(4, 8, 32))
        assert moe.aux_loss_value > 0
        assert float(moe.get_aux_loss().item()) > 0

    def test_domain_load_and_balance(self):
        moe = SpecializedMoEFFN(d_model=32, experts_per_domain=2, top_k=2)
        moe(torch.randn(2, 16, 32))
        loads = moe.domain_load()
        assert set(loads) == set(DEFAULT_DOMAIN_EXPERTS)
        assert sum(loads.values()) > 0
        assert 0.0 <= moe.routing_balance() <= 1.0
        assert len(moe.expert_load()) == moe.num_experts

    def test_add_domain(self):
        moe = SpecializedMoEFFN(d_model=32, experts_per_domain=2, top_k=2)
        before = moe.num_experts
        total = moe.add_domain("assembly")
        assert total == before + 2
        assert moe.domain_of(before) == "assembly"
        out = moe(torch.randn(1, 8, 32))
        assert out.shape == (1, 8, 32)

    def test_custom_domains(self):
        moe = SpecializedMoEFFN(
            d_model=32, expert_types=("geometry", "optimization"), experts_per_domain=1, top_k=2
        )
        assert moe.num_experts == 2
        assert moe.expert_types == ["geometry", "optimization"]

    def test_validation(self):
        with pytest.raises(ValueError):
            SpecializedMoEFFN(d_model=32, expert_types=(), experts_per_domain=1, top_k=1)
        with pytest.raises(ValueError):
            SpecializedMoEFFN(d_model=32, experts_per_domain=1, top_k=6)  # > total
        with pytest.raises(ValueError):
            SpecializedMoEFFN(d_model=32, experts_per_domain=0, top_k=1)

    def test_register_expert_type(self):
        register_expert_type("drafting")
        assert "drafting" in registered_expert_types()
        # Idempotent re-registration.
        register_expert_type("drafting")
        assert registered_expert_types().count("drafting") == 1
        with pytest.raises(ValueError):
            register_expert_type("")

    def test_domain_of_out_of_range(self):
        moe = SpecializedMoEFFN(d_model=32, experts_per_domain=1, top_k=1)
        with pytest.raises(IndexError):
            moe.domain_of(999)
