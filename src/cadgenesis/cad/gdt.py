"""cadgenesis.cad.gdt
===================
Geometric dimensioning and tolerancing (GD&T): geometric tolerances,
datum references and feature control frames.

Supports the standard geometric characteristic symbols grouped by class:
form (flatness, straightness, circularity, cylindricity), orientation
(parallelism, perpendicularity, angularity), location (position, concentricity,
symmetry) and runout (circular runout, total runout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# geometric characteristic symbols by class
GEOMETRIC_TOLERANCES: dict[str, str] = {
    "FLATNESS": "Form",
    "STRAIGHTNESS": "Form",
    "CIRCULARITY": "Form",
    "CYLINDRICITY": "Form",
    "PARALLELISM": "Orientation",
    "PERPENDICULARITY": "Orientation",
    "ANGULARITY": "Orientation",
    "POSITION": "Location",
    "CONCENTRICITY": "Location",
    "SYMMETRY": "Location",
    "CIRCULAR_RUNOUT": "Runout",
    "TOTAL_RUNOUT": "Runout",
    "PROFILE_OF_LINE": "Profile",
    "PROFILE_OF_SURFACE": "Profile",
}

MANUFACTURING_TOLERANCES = (
    "PLUS_MINUS",
    "LIMIT",
    "BASIC",
    "GENERAL",
)

FORM_TOLERANCES = ("FLATNESS", "STRAIGHTNESS", "CIRCULARITY", "CYLINDRICITY")
ORIENTATION_TOLERANCES = ("PARALLELISM", "PERPENDICULARITY", "ANGULARITY")
LOCATION_TOLERANCES = ("POSITION", "CONCENTRICITY", "SYMMETRY")
RUNOUT_TOLERANCES = ("CIRCULAR_RUNOUT", "TOTAL_RUNOUT")


@dataclass
class DatumReference:
    """A single datum in a feature control frame."""

    datum: str  # e.g. "A", "B", "C"
    material_condition: str = ""  # M (maximum), L (least), S (regardless)

    def to_dict(self) -> dict[str, Any]:
        return {"datum": self.datum, "material_condition": self.material_condition}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatumReference:
        return cls(
            datum=str(data["datum"]),
            material_condition=str(data.get("material_condition", "")),
        )


@dataclass
class FeatureControlFrame:
    """A GD&T feature control frame attached to a feature."""

    characteristic: str  # e.g. "POSITION"
    tolerance: float  # geometric tolerance value (mm)
    datums: list[DatumReference] = field(default_factory=list)
    tolerance_zone: str = "diameter"  # "diameter" | "spherical" | "width"
    material_condition: str = ""  # applies to the controlled feature
    feature: str = ""  # reference to the feature being controlled

    def __post_init__(self) -> None:
        if self.characteristic not in GEOMETRIC_TOLERANCES:
            raise ValueError(f"unknown geometric characteristic {self.characteristic!r}")
        if self.tolerance <= 0:
            raise ValueError("geometric tolerance must be positive")

    @property
    def class_name(self) -> str:
        return GEOMETRIC_TOLERANCES[self.characteristic]

    def to_dict(self) -> dict[str, Any]:
        return {
            "characteristic": self.characteristic,
            "tolerance": self.tolerance,
            "datums": [d.to_dict() for d in self.datums],
            "tolerance_zone": self.tolerance_zone,
            "material_condition": self.material_condition,
            "feature": self.feature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureControlFrame:
        return cls(
            characteristic=str(data["characteristic"]),
            tolerance=float(data["tolerance"]),
            datums=[DatumReference.from_dict(d) for d in data.get("datums", [])],
            tolerance_zone=str(data.get("tolerance_zone", "diameter")),
            material_condition=str(data.get("material_condition", "")),
            feature=str(data.get("feature", "")),
        )


@dataclass
class Datum:
    """A datum feature with an identifier and an optional reference face."""

    identifier: str  # e.g. "A"
    feature: str = ""  # reference to a feature / face
    geometry: str = "plane"  # "plane" | "axis" | "point" | "surface"

    def to_dict(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "feature": self.feature, "geometry": self.geometry}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Datum:
        return cls(
            identifier=str(data["identifier"]),
            feature=str(data.get("feature", "")),
            geometry=str(data.get("geometry", "plane")),
        )


@dataclass
class ManufacturingTolerance:
    """A classic manufacturing tolerance (plus/minus, limit or general)."""

    kind: str  # PLUS_MINUS | LIMIT | BASIC | GENERAL
    feature: str = ""
    nominal: float = 0.0
    plus: float = 0.0
    minus: float = 0.0
    upper_limit: float = 0.0
    lower_limit: float = 0.0
    standard: str = ""  # e.g. "ISO 2768-mK"

    def __post_init__(self) -> None:
        if self.kind not in MANUFACTURING_TOLERANCES:
            raise ValueError(
                f"unknown manufacturing tolerance kind {self.kind!r}; "
                f"expected one of {MANUFACTURING_TOLERANCES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "feature": self.feature,
            "nominal": self.nominal,
            "plus": self.plus,
            "minus": self.minus,
            "upper_limit": self.upper_limit,
            "lower_limit": self.lower_limit,
            "standard": self.standard,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManufacturingTolerance:
        return cls(
            kind=str(data["kind"]),
            feature=str(data.get("feature", "")),
            nominal=float(data.get("nominal", 0.0)),
            plus=float(data.get("plus", 0.0)),
            minus=float(data.get("minus", 0.0)),
            upper_limit=float(data.get("upper_limit", 0.0)),
            lower_limit=float(data.get("lower_limit", 0.0)),
            standard=str(data.get("standard", "")),
        )


@dataclass
class GDTSpecification:
    """A part-level GD&T specification: datums, control frames, tolerances."""

    datums: list[Datum] = field(default_factory=list)
    control_frames: list[FeatureControlFrame] = field(default_factory=list)
    manufacturing_tolerances: list[ManufacturingTolerance] = field(default_factory=list)

    def add_datum(self, datum: Datum) -> Datum:
        if any(d.identifier == datum.identifier for d in self.datums):
            raise KeyError(f"datum {datum.identifier!r} already defined")
        self.datums.append(datum)
        return datum

    def add_control_frame(self, frame: FeatureControlFrame) -> FeatureControlFrame:
        self.control_frames.append(frame)
        return frame

    def add_manufacturing_tolerance(
        self, tolerance: ManufacturingTolerance
    ) -> ManufacturingTolerance:
        self.manufacturing_tolerances.append(tolerance)
        return tolerance

    def validate(self) -> list[str]:
        """Check datum references resolve and tolerances are consistent."""
        problems: list[str] = []
        valid_datums = {d.identifier for d in self.datums}
        for frame in self.control_frames:
            problems.extend(
                f"feature control frame {frame.characteristic!r} references "
                f"undefined datum {ref.datum!r}"
                for ref in frame.datums
                if ref.datum not in valid_datums
            )
        for tolerance in self.manufacturing_tolerances:
            if tolerance.kind == "LIMIT" and tolerance.upper_limit < tolerance.lower_limit:
                problems.append(
                    f"limit tolerance on {tolerance.feature or 'feature'!r} has "
                    "upper_limit below lower_limit"
                )
            if tolerance.kind == "PLUS_MINUS" and (tolerance.plus < 0 or tolerance.minus < 0):
                problems.append("plus/minus tolerance values must be non-negative")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "datums": [d.to_dict() for d in self.datums],
            "control_frames": [f.to_dict() for f in self.control_frames],
            "manufacturing_tolerances": [t.to_dict() for t in self.manufacturing_tolerances],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GDTSpecification:
        return cls(
            datums=[Datum.from_dict(d) for d in data.get("datums", [])],
            control_frames=[
                FeatureControlFrame.from_dict(f) for f in data.get("control_frames", [])
            ],
            manufacturing_tolerances=[
                ManufacturingTolerance.from_dict(t)
                for t in data.get("manufacturing_tolerances", [])
            ],
        )


__all__ = [
    "FORM_TOLERANCES",
    "GEOMETRIC_TOLERANCES",
    "LOCATION_TOLERANCES",
    "MANUFACTURING_TOLERANCES",
    "ORIENTATION_TOLERANCES",
    "RUNOUT_TOLERANCES",
    "Datum",
    "DatumReference",
    "FeatureControlFrame",
    "GDTSpecification",
    "ManufacturingTolerance",
]
