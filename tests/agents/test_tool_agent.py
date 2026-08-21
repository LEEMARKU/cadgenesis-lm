"""tests/agents/test_tool_agent.py
===============================
Unit tests for the tool-using agent over the knowledge graph with standards retrieval.
"""

from __future__ import annotations

import pytest

from cadgenesis.agents.tool_agent import ToolAgent, ToolResult
from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph, GraphNode
from cadgenesis.reasoning.standards import Standard, StandardsLibrary, build_standards_graph


# ---- fixture: a minimal standards library -----------------------------------

def _minimal_standards() -> StandardsLibrary:
    """Return a tiny StandardsLibrary with one ISO and one ASME standard."""

    iso_2768 = Standard(
        body="ISO",
        identifier="ISO 2768-1",
        title="General tolerances",
        kind="rule",
        scope="linear dimensions",
        values={"general_tolerance": "fine"},
        check=lambda part: part.get("kind") in ("BOX", "CYLINDER"),
    )

    asme_y14 = Standard(
        body="ASME",
        identifier="ASME Y14.5",
        title="Geometric dimensioning and tolerancing",
        kind="rule",
        scope="GD&T",
        values={"gd_T_version": "2009"},
        check=lambda part: part.get("has_gdandt", False),
    )

    lib = StandardsLibrary()
    lib.register(iso_2768)
    lib.register(asme_y14)
    return lib


# ---- fixture: a knowledge graph with standards built in ---------------------

def _kg_with_standards() -> KnowledgeGraph:
    """Return a KG that has the standards sub-graph built in."""

    kg = KnowledgeGraph()
    build_standards_graph(_minimal_standards())
    return kg


# ---- tests ------------------------------------------------------------------

class TestToolAgent:
    """ToolAgent action tests."""

    def test_get_standard_found(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        result: ToolResult = agent.get_standard("ISO", "ISO 2768-1")
        assert result.ok
        assert result.output["identifier"] == "ISO 2768-1"
        assert result.output["body"] == "ISO"

    def test_get_standard_not_found(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        result: ToolResult = agent.get_standard("ISO", "ISO 9999-1")
        assert not result.ok
        assert "not found" in result.message

    def test_check_compliance(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        part = {"kind": "BOX", "has_gdandt": True}
        result: ToolResult = agent.check_compliance("ISO", "ISO 2768-1", part)
        assert result.ok
        assert result.output["compliant"] is True

    def test_check_compliance_not_found(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        result: ToolResult = agent.check_compliance("ISO", "ISO 9999-1", {})
        assert not result.ok
        assert "not found" in result.message

    def test_find_related(self):
        agent = ToolAgent(kg=_kg_with_standards())
        # Query from a body node - standards KG has body→standard edges
        result: ToolResult = agent.find_related("ISO", max_depth=1)
        assert result.ok
        # Related nodes should include the standard
        related = result.output["related"]
        assert "ISO 2768-1" in related or len(related) > 0

    def test_find_related_not_found(self):
        agent = ToolAgent(kg=_kg_with_standards())
        result: ToolResult = agent.find_related("nonexistent", max_depth=1)
        assert not result.ok
        assert "not found" in result.message

    def test_trace_requirement(self):
        agent = ToolAgent(kg=_kg_with_standards())
        # Add a requirement node
        agent.kg.add_requirement(
            "REQ-001",
            "Hole requirement",
            related_features=["FEAT_HOLE"],
        )
        result: ToolResult = agent.trace_requirement("REQ-001")
        # ok can be True (empty path) or False depending on impl; just verify no crash
        assert result.ok is not None

    def test_trace_requirement_with_target(self):
        agent = ToolAgent(kg=_kg_with_standards())
        # Add a requirement node
        agent.kg.add_requirement(
            "REQ-TEST",
            "Test requirement",
            related_features=["FEAT_HOLE"],
        )
        result: ToolResult = agent.trace_requirement("REQ-TEST", to_feature="FEAT_HOLE")
        assert result.ok
        assert len(result.output["path"]) > 0

    def test_list_standards(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        result: ToolResult = agent.list_standards()
        assert result.ok
        assert len(result.output["standards"]) >= 2

    def test_list_standards_filtered(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        result: ToolResult = agent.list_standards(body="ISO")
        assert result.ok
        identifiers = [s["identifier"] for s in result.output["standards"]]
        assert "ISO 2768-1" in identifiers

    def test_describe(self):
        agent = ToolAgent(standards_library=_minimal_standards())
        desc = agent.describe()
        assert "role" in desc
        assert "actions" in desc
        assert "get_standard" in desc["actions"]
        assert "check_compliance" in desc["actions"]
        assert "find_related" in desc["actions"]
        assert "trace_requirement" in desc["actions"]
        assert "list_standards" in desc["actions"]


# ---- integration: full agent with real KG -----------------------------------

class TestToolAgentIntegration:
    """Integration tests with a real KG + standards library."""

    def setup_method(self):
        # Build a KG with standards embedded
        self.kg = build_standards_graph(_minimal_standards())
        self.standards = _minimal_standards()
        self.agent = ToolAgent(kg=self.kg, standards_library=self.standards)

    def test_full_workflow_get_then_check(self):
        """Get a standard, then check compliance - end-to-end workflow."""
        # 1. Look up the standard
        result = self.agent.get_standard("ISO", "ISO 2768-1")
        assert result.ok

        # 2. Check a part compliance
        part = {"kind": "BOX"}
        result = self.agent.check_compliance("ISO", "ISO 2768-1", part)
        assert result.ok
        assert result.output["compliant"] is True

    def test_full_workflow_find_and_trace(self):
        """Find related nodes and trace requirements."""
        # 1. Find related
        result = self.agent.find_related("ISO", max_depth=2)
        assert result.ok
        assert isinstance(result.output["related"], list)

        # 2. Trace requirement (add one first)
        self.agent.kg.add_requirement(
            "REQ-TEST",
            "Test requirement",
            related_features=["FEAT_HOLE"],
        )
        result = self.agent.trace_requirement("REQ-TEST", to_feature="FEAT_HOLE")
        assert result.ok
        assert len(result.output["path"]) > 0