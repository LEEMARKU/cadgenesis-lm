"""cadgenesis.multimodal.encoders.cad
===================================
CAD-file encoder (STEP, IGES, Parasolid, Fusion360, SolidWorks, FreeCAD,
OpenSCAD).

The encoder normalises any supported CAD source into a :class:`CADDocument` —
a format-tagged feature tree with parameters, materials, constraints and
assembly references — then builds a deterministic structural descriptor that
an MLP maps into the shared raw-feature space.

:func:`parse_cad_file` is a dependency-free, best-effort text parser: it
detects the format (by extension and content signature) and extracts feature
families / entities heuristically so real files can be understood without any
third-party CAD kernel.  When a kernel *is* available (FreeCAD / pythonOCC),
the returned document can be enriched externally before encoding.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder

# Canonical feature families recognised by the structural descriptor.
_FEATURE_FAMILIES: tuple[str, ...] = (
    "extrude",
    "revolve",
    "sweep",
    "loft",
    "fillet",
    "chamfer",
    "hole",
    "pocket",
    "boss",
    "rib",
    "shell",
    "pattern",
    "boolean",
    "draft",
    "thread",
    "cut",
)

_MATERIALS: tuple[str, ...] = (
    "steel",
    "aluminum",
    "titanium",
    "copper",
    "brass",
    "plastic",
    "carbon_fiber",
    "wood",
)


class CADFileFormat(str, Enum):
    """Supported CAD interchange / native formats."""

    STEP = "STEP"
    IGES = "IGES"
    PARASOLID = "PARASOLID"
    FUSION360 = "FUSION360"
    SOLIDWORKS = "SOLIDWORKS"
    FREECAD = "FREECAD"
    OPENSCAD = "OPENSCAD"


_EXTENSION_FORMAT: dict[str, CADFileFormat] = {
    ".step": CADFileFormat.STEP,
    ".stp": CADFileFormat.STEP,
    ".iges": CADFileFormat.IGES,
    ".igs": CADFileFormat.IGES,
    ".x_t": CADFileFormat.PARASOLID,
    ".x_b": CADFileFormat.PARASOLID,
    ".f3d": CADFileFormat.FUSION360,
    ".sldprt": CADFileFormat.SOLIDWORKS,
    ".sldasm": CADFileFormat.SOLIDWORKS,
    ".fcstd": CADFileFormat.FREECAD,
    ".scad": CADFileFormat.OPENSCAD,
}


@dataclass
class CADFeature:
    """A single feature inside a CAD feature tree."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _FEATURE_FAMILIES:
            raise ValueError(
                f"unknown CAD feature kind {self.kind!r}; expected one of {_FEATURE_FAMILIES}"
            )


@dataclass
class CADDocument:
    """Normalised representation of a CAD part / assembly."""

    format: CADFileFormat
    name: str = ""
    features: list[CADFeature] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    materials: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assemblies: list[dict[str, Any]] = field(default_factory=list)
    raw_stats: dict[str, int] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def add_feature(self, kind: str, **params: Any) -> CADFeature:
        feature = CADFeature(kind=kind, params=dict(params))
        self.features.append(feature)
        return feature

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "name": self.name,
            "features": [{"kind": f.kind, "params": dict(f.params)} for f in self.features],
            "parameters": dict(self.parameters),
            "materials": list(self.materials),
            "constraints": list(self.constraints),
            "assemblies": [dict(a) for a in self.assemblies],
            "raw_stats": dict(self.raw_stats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CADDocument:
        """Rebuild a document from :meth:`to_dict` output (round-trip safe)."""
        return cls(
            format=CADFileFormat(data["format"]),
            name=data.get("name", ""),
            features=[
                CADFeature(kind=item["kind"], params=dict(item.get("params", {})))
                for item in data.get("features", [])
            ],
            parameters={k: float(v) for k, v in data.get("parameters", {}).items()},
            materials=list(data.get("materials", [])),
            constraints=list(data.get("constraints", [])),
            assemblies=[dict(a) for a in data.get("assemblies", [])],
            raw_stats={k: int(v) for k, v in data.get("raw_stats", {}).items()},
        )


# --------------------------------------------------------------------------
# Dependency-free file parsing
# --------------------------------------------------------------------------

# Feature family -> signature patterns found inside textual CAD files.
_SIGNATURES: dict[str, tuple[str, ...]] = {
    "extrude": (r"extrud", r"\bBOS\b", r"PROTRUSION"),
    "revolve": (r"revolv", r"\bREV\b"),
    "sweep": (r"\bsweep\b",),
    "loft": (r"\bloft\b",),
    "fillet": (r"fille?t", r"ROUNDED_EDGE"),
    "chamfer": (r"chamfer", r"CHAMFERED_EDGE"),
    "hole": (r"\bhole\b", r"\bHOLES?\b", r"DRILLED"),
    "pocket": (r"pocket", r"\bPOCKET\b"),
    "boss": (r"\bboss\b", r"\bBOSS\b"),
    "rib": (r"\brib\b", r"\bRIB\b"),
    "shell": (r"\bshell\b", r"\bSHELL\b"),
    "pattern": (r"pattern", r"\bPATTERN\b", r"\bARRAY\b", r"\bGRID\b"),
    "boolean": (r"boolean", r"\bUNION\b", r"\bCUT\b", r"\bINTERSECT\b"),
    "draft": (r"draft", r"\bDRAFT\b"),
    "thread": (r"thread", r"SCREW_THREAD"),
    "cut": (r"\bcut\b", r"\bCUTOUT\b", r"REMOVED_MATERIAL"),
}

_RAW_STATS_PATTERNS: dict[str, str] = {
    "cartesian_points": r"CARTESIAN_POINT",
    "edges": r"(?:EDGE_CURVE|EDGE)\b",
    "faces": r"(?:ADVANCED_FACE|FACE)\b",
    "vertices": r"VERTEX_POINT",
    "solids": r"(?:MANIFOLD_SOLID_BREP|SOLID)\b",
    "bsplines": r"B_SPLINE",
    "circles": r"CIRCLE",
    "conics": r"(?:CONIC|PARABOLA|HYPERBOLA)",
}


def detect_format(source: str | Path) -> CADFileFormat:
    """Detect the CAD format from a file extension or content signature."""
    path = Path(source)
    ext = path.suffix.lower()
    if ext in _EXTENSION_FORMAT:
        return _EXTENSION_FORMAT[ext]
    try:
        text = Path(source).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = str(source)
    upper = text[:4096].upper()
    if "ISO-10303-21" in upper or "FILE_DESCRIPTION" in upper:
        return CADFileFormat.STEP
    if upper.startswith("S") and ("IGES" in upper or "IGS" in upper):
        return CADFileFormat.IGES
    if "PARASOLID" in upper:
        return CADFileFormat.PARASOLID
    if "FUSION360" in upper or "ADSKF3D" in upper:
        return CADFileFormat.FUSION360
    if "SOLIDWORKS" in upper or "SLDPRT" in upper:
        return CADFileFormat.SOLIDWORKS
    if "FREECAD" in upper or "FCDOCUMENT" in upper:
        return CADFileFormat.FREECAD
    if "SCAD" in upper or "module " in text[:4096] or "cube(" in text[:4096]:
        return CADFileFormat.OPENSCAD
    return CADFileFormat.STEP


def parse_cad_text(text: str, format: CADFileFormat) -> CADDocument:
    """Parse CAD file text into a :class:`CADDocument` (feature histogram).

    Keyword-based feature detection is a *heuristic* structural summary; it is
    deterministic and sufficient for embedding, retrieval and retrieval-based
    CAD generation.  Full B-Rep reconstruction remains the job of the CAD
    execution backends (FreeCAD / OpenCASCADE).
    """
    lowered = text.lower()
    document = CADDocument(format=format)
    for family, patterns in _SIGNATURES.items():
        matches = sum(len(re.findall(p, lowered)) for p in patterns)
        if matches:
            document.add_feature(family, occurrences=matches)
    for key, pattern in _RAW_STATS_PATTERNS.items():
        document.raw_stats[key] = len(re.findall(pattern, text))
    numeric = [float(m) for m in re.findall(r"-?\d+\.?\d*", text)][:64]
    if numeric:
        document.parameters["mean"] = sum(numeric) / len(numeric)
        document.parameters["count"] = float(len(numeric))
    document.name = Path(".").name
    return document


def parse_cad_file(path: str | Path) -> CADDocument:
    """Parse a CAD file on disk into a :class:`CADDocument`."""
    path = Path(path)
    fmt = detect_format(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    document = parse_cad_text(text, fmt)
    document.name = path.name
    return document


# --------------------------------------------------------------------------
# Structural descriptor + encoder
# --------------------------------------------------------------------------

_DESCRIPTOR_SIZE = (
    len(CADFileFormat)
    + len(_FEATURE_FAMILIES)
    + len(_MATERIALS)
    + 1  # material "other" bucket
    + len(_RAW_STATS_PATTERNS)
    + 8  # parameter / constraint / assembly statistics
)


def _material_index(material: str) -> int:
    for i, candidate in enumerate(_MATERIALS):
        if candidate in material.lower():
            return i
    return len(_MATERIALS)  # "other"


def cad_document_descriptor(document: CADDocument) -> torch.Tensor:
    """Deterministic fixed-size structural descriptor of a CAD document.

    Layout: [format one-hot (7)] + [feature-family histogram (16)] +
    [material one-hot+other (9)] + [raw stats (8)] + [8 statistics].
    """
    vec = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    offset = 0

    vec[offset + list(CADFileFormat).index(document.format)] = 1.0
    offset += len(CADFileFormat)

    for feature in document.features:
        vec[offset + _FEATURE_FAMILIES.index(feature.kind)] += 1.0
    vec[offset : offset + len(_FEATURE_FAMILIES)] /= max(len(document.features), 1)
    offset += len(_FEATURE_FAMILIES)

    for material in document.materials[:4]:
        vec[offset + _material_index(material)] += 1.0
    offset += len(_MATERIALS) + 1  # includes "other"

    for i, key in enumerate(_RAW_STATS_PATTERNS):
        vec[offset + i] = math.log1p(document.raw_stats.get(key, 0))
    offset += len(_RAW_STATS_PATTERNS)

    param_values = [abs(v) for v in document.parameters.values()]
    vec[offset + 0] = math.log1p(len(document.features))
    vec[offset + 1] = math.log1p(len(document.parameters))
    vec[offset + 2] = math.log1p(len(document.constraints))
    vec[offset + 3] = math.log1p(len(document.assemblies))
    vec[offset + 4] = math.log1p(sum(map(len, _FEATURE_FAMILIES)))
    vec[offset + 5] = sum(param_values) if param_values else 0.0
    vec[offset + 6] = (sum(param_values) / len(param_values)) if param_values else 0.0
    vec[offset + 7] = 1.0 if document.name else 0.0
    return vec


class CADEncoder(MultimodalEncoder):
    """Encoder for the ``cad`` modality (any supported CAD file format)."""

    modality: ClassVar[Modality] = Modality.CAD

    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(_DESCRIPTOR_SIZE, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, descriptor_size) -> (B, feature_dim)."""
        if x.dim() != 2 or x.shape[-1] != _DESCRIPTOR_SIZE:
            raise ValueError(
                f"CAD encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        """Accepts a ``CADDocument``, a list of documents, a document dict, a
        path, or a ``(B, descriptor_size)`` tensor.  Returns ``(B, feature_dim)``."""
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, (str, Path)):
            inputs = [parse_cad_file(inputs)]
        elif isinstance(inputs, CADDocument):
            inputs = [inputs]
        elif isinstance(inputs, dict):
            inputs = [CADDocument.from_dict(inputs)]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty CAD batch")
        descriptors = torch.stack([cad_document_descriptor(d) for d in items])
        return self.forward(descriptors)

    def encode_document(self, document: CADDocument) -> torch.Tensor:
        return self.encode([document])


__all__ = [
    "CADDocument",
    "CADEncoder",
    "CADFeature",
    "CADFileFormat",
    "cad_document_descriptor",
    "detect_format",
    "parse_cad_file",
    "parse_cad_text",
]
