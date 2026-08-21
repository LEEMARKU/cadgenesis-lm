"""tests/agents/test_pillar5_infrastructure.py
============================================
Unit tests for Pillar 5 agent infrastructure: AgentBase lifecycle, versioning,
registry, loader, plugins and health monitoring.
"""

from __future__ import annotations

import time

import pytest

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.health import AgentHealthMonitor, AgentHealthStatus
from cadgenesis.agents.infrastructure import (
    AgentBase,
    AgentLifecycleManager,
    AgentMetadata,
    AgentState,
    Capability,
)
from cadgenesis.agents.loader import AgentLoader, AgentLoadError
from cadgenesis.agents.plugins import AgentPlugin, PluginManifest
from cadgenesis.agents.registry import AgentRegistry, RegistryError
from cadgenesis.agents.versioning import AgentVersion


class EchoAgent(AgentBase):
    role = "echo"
    actions = ("echo",)
    version = "1.0.0"
    capabilities = (Capability("echo.cap", "echoes input"),)

    def process(self, request: AgentRequest) -> AgentResult:
        return AgentResult(self.role, request.action, True, dict(request.payload))


class FailAgent(AgentBase):
    role = "fail"
    actions = ("boom",)
    version = "2.0.0"

    def process(self, request: AgentRequest) -> AgentResult:
        raise RuntimeError("kaboom")


# ------------------------------------------------------------------ versioning


def test_version_parse_validate():
    v = AgentVersion.parse("1.4.2")
    assert v.major == 1 and v.minor == 4 and v.patch == 2
    assert AgentVersion(1, 0, 0).is_compatible_with(AgentVersion(1, 9, 0))
    assert not AgentVersion(1, 0, 0).is_compatible_with(AgentVersion(2, 0, 0))


def test_version_rejects_bad():
    with pytest.raises(ValueError):
        AgentVersion.parse("not.a.version")
    with pytest.raises(ValueError):
        AgentVersion.parse("1.0")
    with pytest.raises(ValueError):
        AgentVersion(-1, 0, 0)


# ------------------------------------------------------------- agent lifecycle


def test_agent_base_lifecycle():
    agent = EchoAgent()
    assert agent.state == AgentState.CREATED
    agent.start()
    assert agent.state == AgentState.STARTED
    assert agent.is_alive()
    agent.pause()
    assert agent.state == AgentState.PAUSED
    agent.resume()
    agent.stop()
    assert agent.state == AgentState.STOPPED


def test_agent_base_handle_traps_errors():
    agent = FailAgent()
    result = agent.handle(AgentRequest("fail", "boom", {}))
    assert not result.ok
    assert agent.error_count == 1


def test_agent_base_heartbeat_and_health():
    agent = EchoAgent()
    agent.start()
    agent.heartbeat()
    status = agent.health()
    assert status["alive"] is True
    assert "echo.cap" in agent.capability_names()
    assert agent.has_capability("echo.cap")
    assert agent.describe()["role"] == "echo"


def test_lifecycle_manager():
    manager = AgentLifecycleManager()
    agent = EchoAgent()
    manager.register(agent)
    manager.start_all([agent])
    assert agent.state == AgentState.STARTED
    snapshot = manager.snapshot()
    assert snapshot["echo"]["state"] == "started"
    manager.stop_all([agent])
    assert agent.state == AgentState.STOPPED


def test_agent_metadata():
    meta = AgentMetadata(description="desc", author="me")
    data = meta.to_dict()
    assert data["description"] == "desc"
    assert data["author"] == "me"


# ------------------------------------------------------------------- registry


def test_registry_register_and_get():
    registry = AgentRegistry()
    agent = EchoAgent()
    registry.register(agent)
    assert registry.get("echo") is agent
    assert "echo" in registry
    assert len(registry) == 1


def test_registry_duplicate_raises():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    with pytest.raises(RegistryError):
        registry.register(EchoAgent())


def test_registry_find_by_capability():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    assert [a.role for a in registry.find_by_capability("echo.cap")] == ["echo"]
    assert [a.role for a in registry.find_by_action("echo")] == ["echo"]


def test_registry_require_and_discover():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    assert registry.require("echo") is not None
    with pytest.raises(RegistryError):
        registry.require("missing")
    found = registry.discover(query="echo")
    assert found == ["echo"]


def test_registry_snapshot():
    registry = AgentRegistry()
    registry.register(EchoAgent())
    snapshot = registry.snapshot()
    assert snapshot[0]["role"] == "echo"


# ---------------------------------------------------------------------- loader


def test_loader_load_class():
    loader = AgentLoader()
    agent = loader.load_class("cadgenesis.agents.planner.PlannerAgent")
    assert agent is not None
    assert agent.role == "planner"


def test_loader_scan_package():
    loader = AgentLoader(package="cadgenesis.agents")
    agents = loader.load_module("cadgenesis.agents.planner")
    assert any(a.role == "planner" for a in agents)


def test_loader_bad_module():
    with pytest.raises(AgentLoadError):
        AgentLoader().load_module("cadgenesis.agents.does_not_exist")


def test_loader_instantiates_fleet():
    from cadgenesis.agents.fleet import create_fleet_registry

    registry = create_fleet_registry()
    assert len(registry) == 18


# --------------------------------------------------------------------- plugins


def test_plugin_manifest():
    manifest = PluginManifest(
        name="x", version="1.0.0", description="d", author="a", requires=("a",)
    )
    assert manifest.version_obj == AgentVersion(1, 0, 0)


def test_agent_plugin_lifecycle():
    class MyPlugin(AgentPlugin):
        def create_agents(self):
            return [EchoAgent()]

    plugin = MyPlugin()
    agents = plugin.create_agents()
    assert [a.role for a in agents] == ["echo"]
    plugin.on_install()
    plugin.on_enable()


# ---------------------------------------------------------------------- health


def test_health_monitor_registers_and_checks():
    monitor = AgentHealthMonitor()
    agent = EchoAgent()
    monitor.register(agent)
    agent.start()
    agent.heartbeat()
    result = monitor.check(agent)
    assert result.ok
    summary = monitor.check_all([agent])
    assert summary[0].ok


def test_health_monitor_stale_agent():
    monitor = AgentHealthMonitor(timeout=0.01)
    agent = EchoAgent()
    monitor.register(agent)
    agent.start()
    time.sleep(0.05)
    result = monitor.check(agent)
    assert not result.ok


def test_health_status_to_dict():
    status = AgentHealthStatus("echo", True)
    data = status.to_dict()
    assert data["role"] == "echo"
