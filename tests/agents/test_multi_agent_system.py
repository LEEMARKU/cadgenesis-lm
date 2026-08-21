"""
tests/agents/test_multi_agent_system.py
========================================
Tests for cadgenesis.agents.multi_agent_system.MultiAgentSystem:

- 8 required agent roles are present and trainable
- agent communication bus shape
- shared-memory integration (memory bank conditioning)
- gradient flow through agents + memory
- integration with GeometryAwareTransformer / SelfDesigningTransformer decode
- determinism (same input + eval mode → same output)
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.agents import MultiAgentSystem
from cadgenesis.config import CADConfig
from cadgenesis.memory import LayerIntegratedMemorySystem
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.self_designing import SelfDesigningTransformer


@pytest.fixture
def agents() -> MultiAgentSystem:
    torch.manual_seed(0)
    return MultiAgentSystem(d_model=64)


@pytest.fixture
def mem() -> LayerIntegratedMemorySystem:
    torch.manual_seed(0)
    return LayerIntegratedMemorySystem(d_model=64)


REQUIRED_ROLES = [
    "planner",
    "geometry",
    "constraint",
    "manufacturing",
    "validation",
    "optimization",
    "assembly",
    "simulation",
]


class TestRoles:
    def test_agent_names_match_requirements(self, agents):
        assert agents.agent_names == REQUIRED_ROLES

    def test_eight_agents_exist(self, agents):
        count = sum(1 for n in dir(agents) if n.endswith("_agent"))
        assert count == 8

    def test_roles_are_trainable(self, agents):
        params = [p for p in agents.parameters() if p.requires_grad]
        assert len(params) > 0
        assert all(p.grad is None for p in params)


class TestAgentBus:
    def test_bus_shape(self, agents):
        x = torch.randn(2, 16, 64)
        bus = agents(x)
        assert bus.shape == (2, 16, 64)

    def test_gradient_flows(self, agents, mem):
        x = torch.randn(2, 8, 64)
        bank = mem.get_combined_memory_bank(batch_size=2)
        # Include a memory bank so the memory-conditioning params are exercised.
        bus = agents(x, memory_bank=bank)
        bus.sum().backward()
        for name, param in agents.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_in_eval(self, agents):
        agents.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            a = agents(x)
            b = agents(x)
        assert torch.allclose(a, b)


class TestSharedMemoryIntegration:
    def test_memory_conditioning_changes_output(self, agents, mem):
        x = torch.randn(2, 8, 64)
        bank = mem.get_combined_memory_bank(batch_size=2)
        agents.eval()
        with torch.no_grad():
            no_mem = agents(x)
            with_mem = agents(x, memory_bank=bank)
        assert no_mem.shape == with_mem.shape == (2, 8, 64)
        # memory conditioning must actually influence the agents' output
        assert not torch.allclose(no_mem, with_mem)

    def test_memory_conditioning_gradients(self, agents, mem):
        x = torch.randn(2, 8, 64)
        bank = mem.get_combined_memory_bank(batch_size=2)
        bus = agents(x, memory_bank=bank)
        bus.sum().backward()
        for name, param in agents.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_memory_context_shape(self, agents):
        ctx = agents._memory_context(torch.randn(2, 288, 64))
        assert ctx.shape == (2, 1, 64)


class TestModelIntegration:
    def test_decoder_agents_use_memory(self):
        torch.manual_seed(0)
        cfg = CADConfig.mini()
        model = GeometryAwareTransformer(cfg)
        src = torch.randint(0, 50, (2, 12))
        tgt_in = torch.randint(0, 30, (2, 6))
        tgt_type = torch.randint(0, 3, (2, 6))
        logits, conf = model(src, tgt_in, tgt_type)
        assert logits.shape[1] == 6
        assert conf.shape == (2, 6, 1)
        (logits.sum() + conf.sum()).backward()
        for name, param in model.multi_agent_system.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_self_designing_transformer_forward(self):
        torch.manual_seed(0)
        model = SelfDesigningTransformer(CADConfig.mini())
        src = torch.randint(0, 50, (2, 12))
        tgt_in = torch.randint(0, 30, (2, 6))
        tgt_type = torch.randint(0, 3, (2, 6))
        logits, conf = model(src, tgt_in, tgt_type)
        assert logits.ndim == 3
        assert conf.shape == (2, 6, 1)
