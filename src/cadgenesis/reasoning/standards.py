"""cadgenesis.reasoning.standards
===============================
Engineering standards engine (v6.0, Pillar 7).

A queryable library of engineering standards (ISO, ASME, DIN, ANSI and
company-specific subsets) covering tolerance grades, standard fits, surface
roughness classes, thread pitch series, materials designations and common
design-for-compliance rules.  Provides:

* :class:`Standard` — a single normative entry with body, identifier, scope,
  table data and a compliance predicate.
* :class:`StandardsLibrary` — registers standards by body, resolves lookups
  (``tolerance``, ``fit``, ``roughness``, ``thread``, ``material``, ``rule``)
  and runs compliance checks against part dictionaries.
* :func:`build_standards_graph` — renders the library into a
  :class:`~cadgenesis.reasoning.knowledge_graph.KnowledgeGraph` so standards
  participate in symbolic queries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_STANDARD_BODIES = ("ISO", "ASME", "DIN", "ANSI", "COMPANY")


STANDARD_BODIES = _STANDARD_BODIES


@dataclass
class Standard:
    """A single normative engineering standard entry."""

    body: str
    identifier: str
    title: str = ""
    kind: str = "rule"
    scope: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    check: Callable[[dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if self.body.upper() not in _STANDARD_BODIES:
            raise ValueError(
                f"unknown standards body {self.body!r}; expected one of {_STANDARD_BODIES}"
            )
        if not self.identifier:
            raise ValueError("standard identifier must be non-empty")
        if self.check is not None and not callable(self.check):
            raise TypeError("Standard.check must be callable or None")

    def applies(self, part: dict[str, Any]) -> bool:
        """True when the standard applies to ``part``.

        ``part`` may carry a ``standards`` list (requested bodies); when set,
        only standards from those bodies apply.
        """
        requested = part.get("standards")
        if requested and self.body.upper() not in {r.upper() for r in requested}:
            return False
        kind = part.get("kind")
        return not (kind and self.kind not in ("rule", kind))

    def compliance(self, part: dict[str, Any]) -> bool:
        """Run the compliance predicate (True when no predicate is set)."""
        if self.check is None:
            return True
        return bool(self.check(part))

    def to_dict(self) -> dict[str, Any]:
        return {
            "body": self.body,
            "identifier": self.identifier,
            "title": self.title,
            "kind": self.kind,
            "scope": self.scope,
            "values": dict(self.values),
        }


@dataclass
class StandardsCheck:
    """Result of verifying a part against a standard."""

    standard: Standard
    passed: bool
    detail: str = ""
    recommendation: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "standard": self.standard.identifier,
            "body": self.standard.body,
            "passed": self.passed,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Default normative tables (industry-typical values, educational subset).
# ---------------------------------------------------------------------------

# ISO 286 tolerance grades: nominal-diameter range -> IT grade tolerances (µm).
_ISO_286: dict[str, tuple[float, dict[int, float]]] = {
    "0-3": (0.0, {5: 4, 6: 6, 7: 10, 8: 14, 9: 25, 10: 40, 11: 60, 12: 100}),
    "3-6": (3.0, {5: 5, 6: 8, 7: 12, 8: 18, 9: 30, 10: 48, 11: 75, 12: 120}),
    "6-10": (6.0, {5: 6, 6: 9, 7: 15, 8: 22, 9: 36, 10: 58, 11: 90, 12: 150}),
    "10-18": (10.0, {5: 8, 6: 11, 7: 18, 8: 27, 9: 43, 10: 70, 11: 110, 12: 180}),
    "18-30": (18.0, {5: 9, 6: 13, 7: 21, 8: 33, 9: 52, 10: 84, 11: 130, 12: 210}),
    "30-50": (30.0, {5: 11, 6: 16, 7: 25, 8: 39, 9: 62, 10: 100, 11: 160, 12: 250}),
    "50-80": (50.0, {5: 13, 6: 19, 7: 30, 8: 46, 9: 74, 10: 120, 11: 190, 12: 300}),
    "80-120": (80.0, {5: 15, 6: 22, 7: 35, 8: 54, 9: 87, 10: 140, 11: 220, 12: 350}),
    "120-180": (120.0, {5: 18, 6: 25, 7: 40, 8: 63, 9: 100, 10: 160, 11: 250, 12: 400}),
    "180-250": (180.0, {5: 20, 6: 29, 7: 46, 8: 72, 9: 115, 10: 185, 11: 290, 12: 460}),
    "250-315": (250.0, {5: 23, 6: 32, 7: 52, 8: 81, 9: 130, 10: 210, 11: 320, 12: 520}),
    "315-400": (315.0, {5: 25, 6: 36, 7: 57, 8: 89, 9: 140, 10: 230, 11: 360, 12: 570}),
    "400-500": (400.0, {5: 27, 6: 40, 7: 63, 8: 97, 9: 155, 10: 250, 11: 400, 12: 630}),
}

# ASME B4.1 preferred fits: fit symbol -> (clearance min µm, clearance max µm).
_ASME_FITS: dict[str, tuple[float, float]] = {
    "RC1": (4.0, 18.0),
    "RC2": (8.0, 30.0),
    "RC3": (12.0, 44.0),
    "RC4": (18.0, 62.0),
    "RC5": (28.0, 90.0),
    "LN1": (-18.0, -4.0),
    "LN2": (-24.0, -8.0),
    "LN3": (-36.0, -12.0),
}

# ISO 1302 / 4287 surface roughness grades: Ra µm upper bounds.
_RA_GRADES: dict[str, float] = {
    "N1": 0.025,
    "N2": 0.05,
    "N3": 0.1,
    "N4": 0.2,
    "N5": 0.4,
    "N6": 0.8,
    "N7": 1.6,
    "N8": 3.2,
    "N9": 6.3,
    "N10": 12.5,
    "N11": 25.0,
}

# ISO 261 coarse thread pitch series: nominal diameter -> pitch (mm).
_ISO_THREADS: dict[str, float] = {
    "M2": 0.4,
    "M3": 0.5,
    "M4": 0.7,
    "M5": 0.8,
    "M6": 1.0,
    "M8": 1.25,
    "M10": 1.5,
    "M12": 1.75,
    "M16": 2.0,
    "M20": 2.5,
    "M24": 3.0,
}

# Material designations across bodies (identifier -> family + key properties).
_MATERIALS: dict[str, dict[str, Any]] = {
    "EN AW-6061": {"bodies": ("ISO", "DIN"), "family": "aluminium", "density": 2.7},
    "EN AW-7075": {"bodies": ("ISO", "DIN"), "family": "aluminium", "density": 2.81},
    "AISI 1045": {"bodies": ("ASME", "ANSI"), "family": "steel", "density": 7.85},
    "AISI 316": {"bodies": ("ASME", "ANSI"), "family": "steel", "density": 8.0},
    "ASTM A36": {"bodies": ("ASME", "ANSI"), "family": "steel", "density": 7.85},
    "UNS C17200": {"bodies": ("ASME", "ANSI"), "family": "copper", "density": 8.25},
    "1.4404": {"bodies": ("DIN", "ISO"), "family": "steel", "density": 8.0},
    "PA6": {"bodies": ("ISO", "DIN", "COMPANY"), "family": "polymer", "density": 1.14},
    "POM-C": {"bodies": ("DIN", "COMPANY"), "family": "polymer", "density": 1.41},
}

# Company-specific design rules (generic, configurable).
_COMPANY_RULES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "min_edge_radius_0.2": lambda p: p.get("min_edge_radius", 0.0) >= 0.2,
    "max_depth_ratio_8": lambda p: p.get("depth_ratio", 0.0) <= 8.0,
    "no_sharp_internal_corners": lambda p: p.get("sharp_internal_corners", False) is False,
}


class StandardsLibrary:
    """Registers and queries engineering standards."""

    def __init__(self, standards: list[Standard] | None = None) -> None:
        self._standards: dict[str, Standard] = {}
        if standards:
            for standard in standards:
                self.register(standard)

    # ------------------------------------------------------------- registry

    def register(self, standard: Standard) -> None:
        if standard.identifier in self._standards:
            raise ValueError(f"standard {standard.identifier!r} already registered")
        self._standards[standard.identifier] = standard

    def get(self, identifier: str) -> Standard | None:
        return self._standards.get(identifier)

    def by_body(self, body: str) -> list[Standard]:
        upper = body.upper()
        return sorted(
            (s for s in self._standards.values() if s.body == upper),
            key=lambda s: s.identifier,
        )

    def by_kind(self, kind: str) -> list[Standard]:
        return sorted(
            (s for s in self._standards.values() if s.kind == kind),
            key=lambda s: s.identifier,
        )

    @property
    def identifiers(self) -> list[str]:
        return sorted(self._standards)

    @property
    def bodies(self) -> list[str]:
        return sorted({s.body for s in self._standards.values()})

    def __len__(self) -> int:
        return len(self._standards)

    # ------------------------------------------------------------- lookups

    def tolerance(self, nominal_mm: float, grade: int) -> float | None:
        """ISO 286 tolerance (µm) for a nominal size and IT grade."""
        if nominal_mm < 0:
            raise ValueError("nominal size must be >= 0")
        for lo, grades in _ISO_286.values():
            if nominal_mm >= lo:
                result = grades.get(grade)
        return result

    def fit(self, symbol: str) -> tuple[float, float] | None:
        """ASME B4.1 fit limits (min, max clearance µm) or None."""
        return _ASME_FITS.get(symbol.upper())

    def roughness_grade(self, symbol: str) -> float | None:
        """ISO roughness grade upper Ra bound (µm) or None."""
        return _RA_GRADES.get(symbol.upper())

    def thread_pitch(self, designation: str) -> float | None:
        """ISO 261 coarse pitch (mm) for an M-diameter or None."""
        return _ISO_THREADS.get(designation.upper())

    def material(self, designation: str) -> dict[str, Any] | None:
        """Material properties for a designation across bodies, or None."""
        match = _MATERIALS.get(designation.upper())
        if match is None:
            for name, info in _MATERIALS.items():
                if name.upper() == designation.upper():
                    return info
        return match

    # ------------------------------------------------------------- checking

    def compliance(
        self,
        part: dict[str, Any],
        standards: list[Standard] | None = None,
    ) -> list[StandardsCheck]:
        """Verify ``part`` against (selected) applicable standards."""
        candidates = standards if standards is not None else list(self._standards.values())
        results: list[StandardsCheck] = []
        for standard in candidates:
            if not standard.applies(part):
                continue
            passed = standard.compliance(part)
            detail = (
                f"complies with {standard.identifier}"
                if passed
                else (f"fails {standard.identifier}: {standard.title or 'rule violation'}")
            )
            results.append(
                StandardsCheck(
                    standard,
                    passed,
                    detail=detail,
                    recommendation=(
                        "Adjust the design parameter to the standard's limit." if not passed else ""
                    ),
                )
            )
        return results

    def passed(self, part: dict[str, Any]) -> bool:
        """All applicable standards comply."""
        return all(check.passed for check in self.compliance(part))

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self._standards),
            "bodies": self.bodies,
            "kinds": sorted({s.kind for s in self._standards.values()}),
            "identifiers": self.identifiers,
        }


def _default_standards() -> list[Standard]:
    """Industry-typical default library (ISO/ASME/DIN/ANSI/company subset)."""
    standards: list[Standard] = [
        Standard(
            "ISO",
            "ISO 286-1",
            title="Tolerances on linear sizes",
            kind="tolerance",
            scope="Tolerance grade assignment",
            values={"grades": [5, 6, 7, 8, 9, 10, 11, 12]},
            check=lambda p: (
                p.get("grade") in (5, 6, 7, 8, 9, 10, 11, 12)
                and (p.get("tolerance_um") or 0.0) >= 0.0
            ),
        ),
        Standard(
            "ISO",
            "ISO 261",
            title="ISO metric screw threads — general plan",
            kind="thread",
            scope="Coarse pitch series",
            values={"series": "coarse"},
            check=lambda p: p.get("thread_pitch") is None or p.get("thread_pitch", 0.0) >= 0.25,
        ),
        Standard(
            "ISO",
            "ISO 1302",
            title="Indication of surface texture",
            kind="roughness",
            scope="Ra grade limits",
            values={"grades": "N1-N11"},
            check=lambda p: (p.get("ra_um") or 0.0) >= 0.0,
        ),
        Standard(
            "ASME",
            "ASME B4.1",
            title="Preferred limits and fits",
            kind="fit",
            scope="Clearance/interference fits",
            values={"fits": list(_ASME_FITS)},
            check=lambda p: (p.get("clearance_um") or 0.0) >= -100.0,
        ),
        Standard(
            "ASME",
            "ASME Y14.5",
            title="Dimensioning and tolerancing",
            kind="gd_t",
            scope="GD&T application",
            check=lambda p: bool(p.get("gd_t_datum")) or bool(p.get("tolerance_um")),
        ),
        Standard(
            "DIN",
            "DIN 509",
            title="Undercuts for turning",
            kind="rule",
            scope="Machining relief",
            check=lambda p: p.get("has_undercut_relief", True) is True,
        ),
        Standard(
            "DIN",
            "DIN 16901",
            title="Plastics moulding tolerances",
            kind="tolerance",
            scope="Injection moulded polymer tolerances",
            check=lambda p: (
                (p.get("tolerance_um") or 0.0) >= 50.0 if p.get("family") == "polymer" else True
            ),
        ),
        Standard(
            "ANSI",
            "ANSI B46.1",
            title="Surface texture",
            kind="roughness",
            scope="Ra / Rz limits",
            check=lambda p: (p.get("ra_um") or 0.0) >= 0.0,
        ),
    ]
    for name, rule in _COMPANY_RULES.items():
        standards.append(
            Standard(
                "COMPANY",
                name,
                title="Company design guideline",
                kind="rule",
                scope="Internal design rules",
                check=rule,
            )
        )
    return standards


def default_standards_library() -> StandardsLibrary:
    """A ready-to-use library seeded with the default standard tables."""
    return StandardsLibrary(_default_standards())


def build_standards_graph(library: StandardsLibrary | None = None) -> Any:
    """Render the library into a :class:`KnowledgeGraph`.

    Nodes are standards (``node_type="standard"``) and bodies
    (``node_type="body"``); edges connect body → standard with relation
    ``"defines"``, so queries can traverse standards by body.
    """
    from cadgenesis.reasoning.knowledge_graph import KnowledgeGraph

    library = library or default_standards_library()
    graph = KnowledgeGraph()
    for body in library.bodies:
        graph.add_node(body, label=f"{body} standards", node_type="body")
    for identifier in library.identifiers:
        standard = library.get(identifier)
        if standard is None:
            continue
        graph.add_node(
            identifier,
            label=standard.title or identifier,
            node_type="standard",
            attributes={
                "body": standard.body,
                "kind": standard.kind,
                "scope": standard.scope,
            },
        )
        graph.add_edge(standard.body, identifier, "defines")
    return graph


__all__ = [
    "STANDARD_BODIES",
    "Standard",
    "StandardsCheck",
    "StandardsLibrary",
    "build_standards_graph",
    "default_standards_library",
]
