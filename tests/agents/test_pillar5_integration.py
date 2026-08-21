"""tests/agents/test_pillar5_integration.py
==========================================
Unit tests for the Pillar 5 integration adapters and AgentsConfig.
"""

from __future__ import annotations

from cadgenesis.agents.integration import (
    ContinualLearningHooks,
    MemoryAdapter,
    PlatformIntegrations,
    ReasoningAdapter,
    TokenizerAdapter,
    TransformerAgentAdapter,
    WorldModelAdapter,
)
from cadgenesis.config import AgentsConfig, CADConfig


def test_transformer_adapter_lazy_construction():
    adapter = TransformerAgentAdapter(d_model=64, memory_heads=2, agent_heads=2)
    block = adapter.block
    assert block is not None
    assert adapter.heads()["memory_heads"] == 2


def test_tokenizer_adapter_encodes():
    adapter = TokenizerAdapter()
    tokens = adapter.encode_text("design a bracket")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_world_model_adapter_reason():
    from cadgenesis.world_model import WorldModelSystem

    system = WorldModelSystem()
    system.add_object(
        "block",
        "base",
        {"length": 10.0, "width": 10.0, "height": 5.0},
        material="steel",
    )
    adapter = WorldModelAdapter(system=system)
    result = adapter.reason("mass", limit_kg=100.0)
    assert result.passed is True
    adapter = WorldModelAdapter()
    assert adapter.reason("bogus_capability", x=1)["ok"] is False


def test_memory_adapter():
    adapter = MemoryAdapter()
    adapter.remember("project", "k", "v")
    entry = adapter.recall("project", "k")
    assert entry.content == "v"
    hits = adapter.retrieve("v", top_k=5)
    assert isinstance(hits, list)


def test_reasoning_adapter():
    adapter = ReasoningAdapter()
    plan = adapter.create_plan("design a part")
    assert plan is not None


def test_platform_integrations_status():
    integrations = PlatformIntegrations()
    status = integrations.status()
    assert "tokenizer_ready" in status
    assert "memory_ready" in status


def test_continual_learning_hooks_stub_safe():
    hooks = ContinualLearningHooks()
    assert hooks.record_experience("t", {}) is None
    assert hooks.consolidate(None) is None


def test_agents_config_defaults():
    config = AgentsConfig()
    assert config.enabled
    assert config.workers == 4


def test_cad_config_includes_agents():
    config = CADConfig()
    assert config.agents.enabled
    assert config.agents.heartbeat_timeout == 30.0
