"""cadgenesis.agents.tool_agent
=============================
Tool-using agent over the knowledge graph with engineering standards retrieval
and verified tool calls.

This agent can query the KnowledgeGraph and StandardsLibrary to perform:
- standards lookup by identifier or keyword
- compliance checking against part dictionaries
- requirement traceability paths
- related concept expansion within the KG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph
from cadgenesis.reasoning.standards import (
    StandardsLibrary,
    build_standards_graph,
    default_standards_library,
)


@dataclass
class ToolResult:
    """Result of a tool call executed by the tool agent."""

    tool: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: str | None = None


class ToolAgent:
    """Tool-using agent over the knowledge graph with standards retrieval.

    Actions:
    - ``get_standard``: Look up an engineering standard by identifier
    - ``check_compliance``: Verify a part against a standard
    - ``find_related``: Find related concepts in the knowledge graph
    - ``trace_requirement``: Trace requirement→feature→operation paths
    - ``list_standards``: List available standards bodies/identifiers
    """

    def __init__(
        self,
        kg: KnowledgeGraph | None = None,
        standards_library: StandardsLibrary | None = None,
    ) -> None:
        self.standards = standards_library or default_standards_library()
        # Build the standards sub-graph into a KG
        std_kg = build_standards_graph(self.standards)
        # Merge with any provided KG, or use the standards KG alone
        if kg is None:
            self.kg = std_kg
        else:
            # Merge: add all standards nodes/edges from std_kg into kg
            for node in std_kg.nodes():
                if not kg.has_node(node.id):
                    kg.add_node(
                        node.id,
                        label=node.label,
                        node_type=node.node_type,
                        attributes=dict(node.attributes) if node.attributes else {},
                    )
            for edge_src in std_kg._out:
                for edge in std_kg._out[edge_src]:
                    try:
                        kg.add_edge(edge.source, edge.target, edge.relation, edge.weight)
                    except ValueError:
                        pass  # edge already exists
            self.kg = kg

    # ------------------------------------------------------------------ actions

    def get_standard(self, body: str, identifier: str) -> ToolResult:
        """Look up an engineering standard by standards body and identifier.

        Parameters
        ----------
        body : str
            Standards body (``"ISO"``, ``"ASME"``, ``"DIN"``, ``"ANSI"``, ``"COMPANY"``).
        identifier : str
            Standard identifier (e.g. ``"ISO 2768-1"``, ``"ASME Y14.5"``).

        Returns
        -------
        ToolResult
            ``ok=True`` with the standard's ``to_dict()`` output, or ``ok=False``
            with an error message.
        """
        std = self.standards.get(identifier)
        if std is None or std.body != body.upper():
            return ToolResult(
                tool="get_standard",
                ok=False,
                message=f"standard {body} {identifier!r} not found",
                error=f"standard {body} {identifier!r} not found",
            )
        return ToolResult(
            tool="get_standard",
            ok=True,
            output=std.to_dict(),
        )

    def check_compliance(
        self, body: str, identifier: str, part: dict[str, Any]
    ) -> ToolResult:
        """Verify a part dictionary against an engineering standard.

        Parameters
        ----------
        body : str
            Standards body (``"ISO"``, ``"ASME"``, ``"DIN"``, ``"ANSI"``, ``"COMPANY"``).
        identifier : str
            Standard identifier.
        part : dict
            Part dictionary describing the CAD model/feature.

        Returns
        -------
        ToolResult
            ``ok=True`` with a compliance result, or ``ok=False`` on error.
        """
        std = self.standards.get(identifier)
        if std is None or std.body != body.upper():
            return ToolResult(
                tool="check_compliance",
                ok=False,
                message=f"standard {body} {identifier!r} not found",
                error=f"standard {body} {identifier!r} not found",
            )
        compliant = std.compliance(part)
        return ToolResult(
            tool="check_compliance",
            ok=True,
            output={
                "compliant": compliant,
                "standard_id": std.identifier,
                "standard_body": std.body,
                "detail": std.to_dict(),
            },
        )

    def find_related(self, node_id: str, max_depth: int = 2) -> ToolResult:
        """Find related concepts in the knowledge graph within *max_depth* hops.

        Parameters
        ----------
        node_id : str
            Node ID to start from.
        max_depth : int
            Maximum traversal depth (default 2).

        Returns
        -------
        ToolResult
            ``ok=True`` with a set of related node IDs, or ``ok=False`` if node
            not found.
        """
        if node_id not in self.kg._nodes:
            return ToolResult(
                tool="find_related",
                ok=False,
                message=f"node {node_id!r} not found in knowledge graph",
                error=f"node {node_id!r} not found in knowledge graph",
            )
        related = self.kg.find_related(node_id, max_depth)
        return ToolResult(
            tool="find_related",
            ok=True,
            output={"related": list(related), "start": node_id, "depth": max_depth},
        )

    def trace_requirement(
        self,
        from_req: str,
        to_feature: str | None = None,
        to_op: str | None = None,
    ) -> ToolResult:
        """Trace from a requirement to related features or operations.

        Parameters
        ----------
        from_req : str
            Requirement ID to start from (e.g. ``"REQ-001"``).
        to_feature : str
            Optional feature ID to trace to.
        to_op : str
            Optional operation ID to trace to.

        Returns
        -------
        ToolResult
            ``ok=True`` with the trace path, or ``ok=False`` if the starting
            requirement is not found.
        """
        path = self.kg.requirement_traceability_path(from_req, to_feature, to_op)
        if from_req not in self.kg._nodes:
            return ToolResult(
                tool="trace_requirement",
                ok=False,
                message=f"requirement {from_req!r} not found in knowledge graph",
                error=f"requirement {from_req!r} not found in knowledge graph",
            )
        return ToolResult(
            tool="trace_requirement",
            ok=True,
            output={"path": path, "start": from_req},
        )

    def list_standards(self, body: str | None = None) -> ToolResult:
        """List available standards from the standards library.

        Parameters
        ----------
        body : str
            Optional standards body filter (``"ISO"``, ``"ASME"``, etc.).

        Returns
        -------
        ToolResult
            ``ok=True`` with a list of standard identifiers, or ``ok=False`` on error.
        """
        # StandardsLibrary stores standards in self._standards dict
        all_stds = self.standards._standards.values() if hasattr(self.standards, "_standards") else []
        if body:
            stds = [s for s in all_stds if s.body.upper() == body.upper()]
        else:
            stds = list(all_stds)
        return ToolResult(
            tool="list_standards",
            ok=True,
            output={
                "standards": [
                    {"body": s.body, "identifier": s.identifier, "title": s.title}
                    for s in stds
                ]
            },
        )

    # ------------------------------------------------------------------ dispatch

    def process(self, action: str, payload: dict[str, Any]) -> ToolResult:
        """Process a tool agent action.

        Supported actions
        -----------------
        - ``get_standard``: payload = {\"body\": \"ISO\", \"identifier\": \"2768-1\"}
        - ``check_compliance``: payload = {\"body\": \"ISO\", \"identifier\": \"2768-1\", \"part\": {...}}
        - ``find_related``: payload = {\"node_id\": \"aluminum\", \"max_depth\": 2}
        - ``trace_requirement``: payload = {\"from_req\": \"REQ-001\", \"to_feature\": \"FEAT_HOLE\"}
        - ``list_standards``: payload = {\"body\": \"ISO\"}  (optional)
        """
        action = action.lower()
        if action == "get_standard":
            return self.get_standard(payload["body"], payload["identifier"])
        if action == "check_compliance":
            return self.check_compliance(
                payload["body"], payload["identifier"], payload["part"]
            )
        if action == "find_related":
            return self.find_related(payload["node_id"], payload.get("max_depth", 2))
        if action == "trace_requirement":
            return self.trace_requirement(
                payload["from_req"],
                payload.get("to_feature"),
                payload.get("to_op"),
            )
        if action == "list_standards":
            return self.list_standards(payload.get("body"))
        return ToolResult(
            tool=action,
            ok=False,
            message=f"unknown action {action!r}",
            error=f"unknown action {action!r}",
        )

    def describe(self) -> dict[str, Any]:
        """Return a description of available actions and their payload schemas."""
        return {
            "role": "tool_agent",
            "actions": {
                "get_standard": {
                    "body": "str (ISO|ASME|DIN|ANSI|COMPANY)",
                    "identifier": "str",
                },
                "check_compliance": {
                    "body": "str (ISO|ASME|DIN|ANSI|COMPANY)",
                    "identifier": "str",
                    "part": "dict (part description)",
                },
                "find_related": {
                    "node_id": "str (KG node ID)",
                    "max_depth": "int (default 2)",
                },
                "trace_requirement": {
                    "from_req": "str (requirement ID)",
                    "to_feature": "str (optional feature ID)",
                    "to_op": "str (optional operation ID)",
                },
                "list_standards": {
                    "body": "str (optional filter: ISO|ASME|DIN|ANSI|COMPANY)",
                },
            },
        }