"""cadgenesis.tools
=================
Tool calling: schemas, registry, executor (bound to the real CAD
execution backends) and the agent-side bridge.
"""

from cadgenesis.tools.agent import AgentToolBridge
from cadgenesis.tools.executor import ToolExecutor
from cadgenesis.tools.registry import ToolRegistry
from cadgenesis.tools.schema import (
    PARAM_TYPES,
    ParameterSpec,
    Permission,
    ToolCall,
    ToolDefinition,
    ToolResult,
    permission_allows,
)

__all__ = [
    "PARAM_TYPES",
    "AgentToolBridge",
    "ParameterSpec",
    "Permission",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "permission_allows",
]
