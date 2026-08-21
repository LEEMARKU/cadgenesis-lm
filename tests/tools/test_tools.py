"""Tests for the M7 tool-calling family: schemas, registry, executor and
the agent bridge."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from cadgenesis.agents.base import AgentRequest
from cadgenesis.tools import (
    AgentToolBridge,
    ParameterSpec,
    Permission,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
)
from cadgenesis.tools.schema import permission_allows

BOX = ["BOX", "NUM_80", "NUM_40", "NUM_20"]
CYL = ["CYLINDER", "NUM_30", "NUM_60"]


def _tool() -> ToolExecutor:
    return ToolExecutor()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_permission_order() -> None:
    assert permission_allows(Permission.READ, Permission.READ)
    assert permission_allows(Permission.EXECUTE, Permission.EXECUTE)
    assert permission_allows(Permission.EXECUTE, Permission.ADMIN)
    assert not permission_allows(Permission.ADMIN, Permission.EXECUTE)
    assert not permission_allows(Permission.EXECUTE, Permission.READ)


def test_parameter_spec_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="type must be one of"):
        ParameterSpec("x", "bogus")


def test_tool_definition_dict_shape() -> None:
    definition = ToolDefinition(
        name="echo",
        description="echoes",
        parameters=(ParameterSpec("msg", "string"),),
        handler=lambda args: args,
    )
    data = definition.to_dict()
    assert data["name"] == "echo"
    assert data["parameters"][0]["type"] == "string"
    assert data["permission"] == "execute"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_lookup_and_list() -> None:
    registry = ToolRegistry()
    definition = ToolDefinition(name="echo", description="echoes", handler=lambda args: args)
    registry.register(definition)
    assert registry.get("echo") is definition
    assert registry.names() == ["echo"]
    assert registry.list_tools()[0]["name"] == "echo"


def test_register_duplicate_rejected() -> None:
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="echo", description="d", handler=lambda a: a))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ToolDefinition(name="echo", description="d", handler=lambda a: a))
    registry.register(
        ToolDefinition(name="echo", description="d", handler=lambda a: a), replace=True
    )
    assert registry.names() == ["echo"]


def test_register_invalid_name_and_handler() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="invalid tool name"):
        registry.register(ToolDefinition(name="not valid!", description="d", handler=lambda a: a))
    with pytest.raises(TypeError, match="handler must be callable"):
        registry.register(ToolDefinition(name="x", description="d", handler="nope"))  # type: ignore[arg-type]


def test_register_duplicate_parameter_names_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="duplicate parameter names"):
        registry.register(
            ToolDefinition(
                name="dup",
                description="d",
                parameters=(ParameterSpec("a", "string"), ParameterSpec("a", "string")),
                handler=lambda a: a,
            )
        )


def test_unregister() -> None:
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="echo", description="d", handler=lambda a: a))
    assert registry.unregister("echo")
    assert not registry.unregister("echo")


def test_validate_call_unknown_tool() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="unknown tool"):
        registry.validate_call(ToolCall(name="ghost"))


def test_validate_call_missing_and_unknown_parameters() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="d",
            parameters=(ParameterSpec("msg", "string"),),
            handler=lambda a: a,
        )
    )
    with pytest.raises(ValueError, match="missing required parameter"):
        registry.validate_call(ToolCall(name="echo"))
    with pytest.raises(ValueError, match="unknown parameter"):
        registry.validate_call(ToolCall(name="echo", arguments={"msg": "hi", "x": 1}))


def test_validate_call_type_checking_and_defaults() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="d",
            parameters=(
                ParameterSpec("msg", "string"),
                ParameterSpec("n", "number", required=False, default=7),
            ),
            handler=lambda a: a,
        )
    )
    with pytest.raises(ValueError, match="must be string"):
        registry.validate_call(ToolCall(name="echo", arguments={"msg": 5}))  # type: ignore[arg-type]
    _, args = registry.validate_call(ToolCall(name="echo", arguments={"msg": "hi"}))
    assert args == {"msg": "hi", "n": 7}


def test_validate_call_permission_enforcement() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="admin_tool",
            description="d",
            permission=Permission.ADMIN,
            handler=lambda a: a,
        )
    )
    with pytest.raises(PermissionError, match="requires permission"):
        registry.validate_call(ToolCall(name="admin_tool"), granted=Permission.READ)
    definition, _ = registry.validate_call(ToolCall(name="admin_tool"), granted=Permission.ADMIN)
    assert definition.name == "admin_tool"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def test_executor_registers_builtins() -> None:
    names = _tool().registry.names()
    assert "validate_program" in names
    assert "execute_program" in names
    assert "analyze_brep" in names
    assert "estimate_cost" in names
    assert "manufacturing_check" in names
    assert "export_program" in names


def test_validate_program_tool_ok_and_fail() -> None:
    tool = _tool()
    ok = tool.dispatch(ToolCall(name="validate_program", arguments={"program": BOX}))
    assert ok.ok
    assert ok.output["all_passed"] is True
    assert ok.output["checks"]
    bad = tool.dispatch(ToolCall(name="validate_program", arguments={"program": ["NO_SUCH_TOK"]}))
    assert bad.ok
    assert bad.output["all_passed"] is False


def test_execute_program_both_backends() -> None:
    tool = _tool()
    for backend in ("freecad", "opencascade"):
        result = tool.dispatch(
            ToolCall(name="execute_program", arguments={"program": BOX, "backend": backend})
        )
        assert result.ok, result.error
        assert result.output["status"] == "ok"
        assert result.output["volume_mm3"] > 0


def test_execute_program_rejects_unknown_backend() -> None:
    result = _tool().dispatch(
        ToolCall(name="execute_program", arguments={"program": BOX, "backend": "nurbs"})
    )
    assert not result.ok
    assert "backend" in result.error


def test_analyze_brep_tool() -> None:
    result = _tool().dispatch(ToolCall(name="analyze_brep", arguments={"program": BOX}))
    assert result.ok, result.error
    assert result.output["all_passed"] is True
    assert result.output["volume_mm3"] > 0


def test_cost_tool() -> None:
    part = json.dumps(
        {
            "material": {"name": "steel", "density_kg_m3": 7850.0},
            "volume_mm3": 50_000.0,
            "processes": ["machining"],
        }
    )
    result = _tool().dispatch(ToolCall(name="estimate_cost", arguments={"part": part}))
    assert result.ok, result.error
    assert result.output["total_usd"] > 0


def test_cost_tool_rejects_bad_json() -> None:
    result = _tool().dispatch(ToolCall(name="estimate_cost", arguments={"part": "not json"}))
    assert not result.ok
    assert "not valid JSON" in result.error


def test_manufacturing_check_tool() -> None:
    thin = json.dumps(
        {
            "material": "steel",
            "processes": ["machining"],
            "min_wall_thickness": 0.2,
        }
    )
    result = _tool().dispatch(ToolCall(name="manufacturing_check", arguments={"part": thin}))
    assert result.ok, result.error
    assert result.output["all_passed"] is False
    assert result.output["max_severity"] >= 3


def test_export_program_tool_writes_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "tool_export.obj")
        result = _tool().dispatch(
            ToolCall(
                name="export_program",
                arguments={"program": BOX, "path": out, "format": "obj"},
                caller="design",
            ),
            granted=Permission.ADMIN,
        )
        assert result.ok, result.error
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


def test_dispatch_permission_denied() -> None:
    result = _tool().dispatch(
        ToolCall(name="execute_program", arguments={"program": BOX}),
        granted=Permission.READ,
    )
    assert not result.ok
    assert "requires permission" in result.error


def test_dispatch_unknown_tool_is_result_not_raise() -> None:
    result = _tool().dispatch(ToolCall(name="ghost"))
    assert not result.ok
    assert "unknown tool" in result.error


# ---------------------------------------------------------------------------
# Agent bridge
# ---------------------------------------------------------------------------


def test_bridge_successful_call() -> None:
    bridge = AgentToolBridge()
    request = AgentRequest(
        role="geometry",
        action="tool_call",
        payload={"tool": "validate_program", "arguments": {"program": BOX}},
    )
    result = bridge.handle(request)
    assert result.ok
    assert result.output["output"]["all_passed"] is True
    assert result.message == "tool 'validate_program' succeeded"


def test_bridge_permission_in_payload() -> None:
    bridge = AgentToolBridge()
    request = AgentRequest(
        role="user",
        action="tool_call",
        payload={
            "tool": "export_program",
            "arguments": {"program": BOX, "path": "nope.obj"},
            "permission": "read",
        },
    )
    result = bridge.handle(request)
    assert not result.ok
    assert "requires permission" in result.message


def test_bridge_missing_tool_name() -> None:
    bridge = AgentToolBridge()
    request = AgentRequest(role="geometry", action="tool_call", payload={})
    result = bridge.handle(request)
    assert not result.ok
    assert "requires a 'tool'" in result.message


def test_bridge_describe_lists_tools() -> None:
    bridge = AgentToolBridge()
    desc = bridge.describe()
    assert desc["action"] == "tool_call"
    assert any(t["name"] == "execute_program" for t in desc["tools"])


def test_bridge_can_handle() -> None:
    assert AgentToolBridge().can_handle("tool_call")
    assert not AgentToolBridge().can_handle("other")


def test_cost_estimate_requires_part_dict() -> None:
    result = _tool().dispatch(ToolCall(name="estimate_cost", arguments={"part": "[1,2]"}))
    assert not result.ok
    assert "expected a JSON object" in result.error


def test_tool_call_caller_propagates() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="d",
            parameters=(ParameterSpec("msg", "string"),),
            handler=lambda a: a,
        )
    )
    call = ToolCall(name="echo", arguments={"msg": "x"}, caller="design")
    result = ToolExecutor(registry).dispatch(call)
    assert result.ok
    assert result.output == {"msg": "x"}
