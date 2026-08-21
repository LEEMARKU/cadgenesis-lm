"""cadgenesis.agents.documentation
================================
Documentation agent.

Produces human-readable engineering summaries and structured reports from the
agent fleet's outputs — the documentation and knowledge-sharing agent of the
Pillar 5 fleet.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability


class DocumentationAgent(AgentBase):
    """Summarizes results and generates engineering documentation."""

    role = "documentation"
    actions = ("summarize", "generate_report")
    version = "1.0.0"
    capabilities = (
        Capability("docs.summarize", "produce a markdown summary of data"),
        Capability("docs.report", "generate a structured engineering report"),
    )

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        if request.action == "summarize":
            return self._ok(
                request,
                {"markdown": self._summarize(payload)},
                "summary generated",
            )
        if request.action == "generate_report":
            title = str(payload.get("title", "Engineering Report"))
            sections = payload.get("sections", {})
            return self._ok(
                request,
                {"report": self._report(title, sections)},
                "report generated",
            )
        return self._fail(request, f"unsupported action {request.action!r}")

    @staticmethod
    def _summarize(payload: dict[str, Any]) -> str:
        lines: list[str] = []
        for key, value in payload.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines) if lines else "_no data provided_"

    @staticmethod
    def _report(title: str, sections: dict[str, Any]) -> dict[str, Any]:
        body: list[str] = [f"# {title}", ""]
        if isinstance(sections, dict):
            for heading, content in sections.items():
                body.append(f"## {heading}")
                body.append("")
                body.append(str(content))
                body.append("")
        return {
            "title": title,
            "sections": list(sections) if isinstance(sections, dict) else [],
            "markdown": "\n".join(body),
            "generated": True,
        }

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
