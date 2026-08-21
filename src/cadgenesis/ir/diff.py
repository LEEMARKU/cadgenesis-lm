"""
cadgenesis.ir.diff
=================
Structural, op-level CAD program diff (v6.4).

Compares two :class:`~cadgenesis.ir.program.CadProgram` revisions and reports
added / removed / changed operations.  Ops are anchored by their
``(position, kind)`` slot — the same authoring slot in two revisions is the
same logical feature — and ``changed`` carries the per-parameter delta.
Content-hash ``op_id``s are NOT the anchor (any parameter change would
otherwise look like add+remove).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.ir.program import CadProgram


def _anchored(program: CadProgram) -> dict[tuple[int, str], dict[str, Any]]:
    """Map ``(position, kind)`` -> op dict for the diff algorithm."""
    return {(step.position, step.kind): step.to_dict() for step in program.steps}


@dataclass
class IrDiffReport:
    """Op-level difference between two program revisions."""

    before_id: str
    after_id: str
    added: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    changed: list[dict[str, Any]] = field(default_factory=list)
    unchanged: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def summary(self) -> dict[str, Any]:
        return {
            "before_id": self.before_id,
            "after_id": self.after_id,
            "added": len(self.added),
            "removed": len(self.removed),
            "changed": len(self.changed),
            "unchanged": self.unchanged,
            "has_changes": self.has_changes,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.summary()
        data["added_ops"] = self.added
        data["removed_ops"] = self.removed
        data["changed_ops"] = self.changed
        return data


def ir_diff(before: CadProgram, after: CadProgram) -> IrDiffReport:
    """Structural diff anchored by ``(position, kind)``.

    - ``added``: ops present in ``after`` at slots absent in ``before``;
    - ``removed``: ops present in ``before`` at slots absent in ``after``;
    - ``changed``: same slot, same kind, differing parameters (with
      ``changed_params`` listing the keys whose values differ);
    - ``unchanged``: identical slots (same kind + params).
    """
    before_ops = _anchored(before)
    after_ops = _anchored(after)
    report = IrDiffReport(
        before_id=before.program_id,
        after_id=after.program_id,
    )

    before_slots = set(before_ops)
    after_slots = set(after_ops)

    report.added = [after_ops[slot] for slot in sorted(after_slots - before_slots)]
    report.removed = [before_ops[slot] for slot in sorted(before_slots - after_slots)]

    for slot in sorted(after_slots & before_slots):
        b = before_ops[slot]
        a = after_ops[slot]
        if b == a:
            report.unchanged += 1
            continue
        b_params = b.get("params", {})
        a_params = a.get("params", {})
        changed_keys = sorted(
            k for k in set(b_params) | set(a_params) if b_params.get(k) != a_params.get(k)
        )
        report.changed.append(
            {
                "position": slot[0],
                "kind": slot[1],
                "before_params": b_params,
                "after_params": a_params,
                "changed_params": changed_keys,
            }
        )
    return report


__all__ = ["IrDiffReport", "ir_diff"]