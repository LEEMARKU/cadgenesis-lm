"""
tests/tools/test_timeout_provenance.py
======================================
Timeout enforcement and provenance tracking for tool calls
(pre-training gate: tool-calling protocol).
"""

from __future__ import annotations

import time

import pytest

from cadgenesis.tools import (
    ParameterSpec,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
)


def _slow_tool(timeout: float | None = 0.05) -> ToolExecutor:
    registry = ToolRegistry()

    def slow(args):
        time.sleep(10.0)
        return {"done": True}

    registry.register(
        ToolDefinition(
            name="slow_tool",
            description="always sleeps",
            parameters=(ParameterSpec("msg", "string"),),
            handler=slow,
            timeout_seconds=timeout,
        )
    )
    return ToolExecutor(registry)


def _fast_tool(timeout: float | None = None) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="echoes",
            parameters=(ParameterSpec("msg", "string"),),
            handler=lambda args: {"echo": args["msg"]},
            timeout_seconds=timeout,
        )
    )
    return ToolExecutor(registry)


class TestTimeout:
    def test_timeout_returns_error_result(self):
        tool = _slow_tool(timeout=0.05)
        started = time.time()
        result = tool.dispatch(ToolCall(name="slow_tool", arguments={"msg": "x"}))
        elapsed = time.time() - started
        assert result.ok is False
        assert "timeout" in result.error
        assert elapsed < 5.0

    def test_no_timeout_runs_fine(self):
        tool = _fast_tool(timeout=5.0)
        result = tool.dispatch(ToolCall(name="echo", arguments={"msg": "hi"}))
        assert result.ok
        assert result.output == {"echo": "hi"}

    def test_no_timeout_configured_runs_fine(self):
        tool = _fast_tool()
        result = tool.dispatch(ToolCall(name="echo", arguments={"msg": "hi"}))
        assert result.ok

    def test_timeout_in_schema_dict(self):
        definition = _fast_tool(timeout=3.0).registry.get("echo")
        assert definition.timeout_seconds == 3.0
        assert definition.to_dict()["timeout_seconds"] == 3.0

    def test_validation_errors_bypass_timeout_thread(self):
        tool = _slow_tool(timeout=0.05)
        result = tool.dispatch(ToolCall(name="ghost"))
        assert result.ok is False
        assert "unknown tool" in result.error


class TestProvenance:
    def test_result_has_call_id_timestamp_duration(self):
        tool = _fast_tool()
        before = time.time()
        result = tool.dispatch(ToolCall(name="echo", arguments={"msg": "hi"}, caller="test"))
        assert result.ok
        assert result.call_id
        assert result.caller == "test"
        assert result.timestamp >= before
        assert result.duration_seconds >= 0.0

    def test_explicit_call_id_preserved(self):
        tool = _fast_tool()
        result = tool.dispatch(
            ToolCall(name="echo", arguments={"msg": "hi"}, call_id="call-abc")
        )
        assert result.call_id == "call-abc"

    def test_run_id_stamped(self):
        tool = _fast_tool()
        result = tool.dispatch(
            ToolCall(name="echo", arguments={"msg": "hi"}), run_id="run-42"
        )
        assert result.run_id == "run-42"

    def test_error_results_carry_provenance(self):
        tool = _fast_tool()
        result = tool.dispatch(
            ToolCall(name="echo", arguments={"msg": 5}, caller="design"),  # type: ignore[arg-type]
            run_id="run-7",
        )
        assert result.ok is False
        assert result.call_id
        assert result.caller == "design"
        assert result.run_id == "run-7"

    def test_timeout_results_carry_provenance(self):
        tool = _slow_tool(timeout=0.05)
        result = tool.dispatch(
            ToolCall(name="slow_tool", arguments={"msg": "x"}, caller="agent"),
            run_id="run-timeout",
        )
        assert result.ok is False
        assert result.caller == "agent"
        assert result.run_id == "run-timeout"
        assert result.call_id

    def test_unique_call_ids(self):
        tool = _fast_tool()
        ids = {
            tool.dispatch(ToolCall(name="echo", arguments={"msg": "hi"})).call_id
            for _ in range(20)
        }
        assert len(ids) == 20

    def test_to_dict_includes_provenance(self):
        tool = _fast_tool()
        result = tool.dispatch(ToolCall(name="echo", arguments={"msg": "hi"}), run_id="r1")
        data = result.to_dict()
        assert "call_id" in data
        assert "run_id" in data
        assert "duration_seconds" in data
        assert data["run_id"] == "r1"