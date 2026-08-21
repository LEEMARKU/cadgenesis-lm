"""cadgenesis.cad.integration.tokenizer_bridge
============================================
Map CAD objects (feature trees, primitives, materials, assemblies, GD&T,
manufacturing) to CAD token strings and full ``CADTokenSequence`` examples
via the existing ``AutonomousCADTokenizer``.

The bridge is *lossy by design*: CAD token sequences are a discrete,
quantized projection of the continuous design.  Structural information is
preserved as token order; numeric parameters are quantized through the
tokenizer's ``NumericTokenizer``.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.features.base import FeatureType

#: FeatureType -> tokenizer feature token string.
FEATURE_TOKEN_MAP: dict[FeatureType, str] = {
    FeatureType.EXTRUDE: "FEAT_EXTRUDE",
    FeatureType.REVOLVE: "FEAT_REVOLVE",
    FeatureType.SWEEP: "FEAT_SWEEP",
    FeatureType.LOFT: "FEAT_LOFT",
    FeatureType.FILLET: "FEAT_FILLET",
    FeatureType.CHAMFER: "FEAT_CHAMFER",
    FeatureType.SHELL: "FEAT_SHELL",
    FeatureType.HOLE: "FEAT_HOLE",
    FeatureType.LINEAR_PATTERN: "FEAT_PATTERN_LIN",
    FeatureType.CIRCULAR_PATTERN: "FEAT_PATTERN_CIRC",
    FeatureType.MIRROR: "FEAT_PATTERN_MIRROR",
    FeatureType.BOOLEAN_UNION: "FEAT_BOOL_UNION",
    FeatureType.BOOLEAN_SUBTRACT: "FEAT_BOOL_CUT",
    FeatureType.BOOLEAN_INTERSECT: "FEAT_BOOL_INTERSECT",
}

#: Material name -> tokenizer material token string (best-effort).
MATERIAL_TOKEN_MAP: dict[str, str] = {
    "AISI 1018": "MAT_STEEL_MILD",
    "AISI 1045": "MAT_STEEL_MEDIUM",
    "AISI 304": "MAT_STEEL_SS_304",
    "AISI 316": "MAT_STEEL_SS_316",
    "6061-T6": "MAT_AL_6061",
    "7075-T6": "MAT_AL_7075",
    "Ti-6Al-4V": "MAT_TI_6AL4V",
    "ABS": "MAT_ABS",
    "PLA": "MAT_PLA",
    "Nylon PA6": "MAT_NYLON_PA6",
    "PEEK": "MAT_PEEK",
    "Polycarbonate": "MAT_POLYCARBONATE",
    "Copper": "MAT_COPPER",
}


class TokenizerBridge:
    """Converts CAD objects into CAD token strings / sequences.

    ``tokenizer`` is an ``AutonomousCADTokenizer`` (or any object exposing
    ``encode_cad_token``, ``encode_length``, ``encode_angle`` and ``vocab``).
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    # -- primitives -----------------------------------------------------------

    def primitive_tokens(self, primitive: Any) -> list[str]:
        """Tokenize a primitive object or ``{"kind": ..., "dims": ...}`` dict."""
        if isinstance(primitive, dict):
            data = primitive
        elif hasattr(primitive, "to_dict"):
            data = primitive.to_dict()
        else:
            raise TypeError(
                f"unsupported primitive type {type(primitive).__name__}; "
                "expected dict or to_dict() object"
            )
        kind = data.get("kind", "").upper()
        prim_token = _primitive_kind_token(kind)
        if prim_token is None:
            raise ValueError(f"no token for primitive kind {kind!r}")
        tokens = [prim_token]
        dims = data.get("dims", {})
        for value in dims.values():
            _bin_idx, num_token = self.tokenizer.encode_length(float(value))
            tokens.append(num_token)
        return tokens

    # -- features -------------------------------------------------------------

    def feature_tokens(self, feature: Any) -> list[str]:
        """Tokenize a feature object or ``{"type": ..., "params": ...}`` dict."""
        if isinstance(feature, dict):
            type_key = feature.get("type")
            params = feature.get("params", {})
        else:
            type_key = getattr(feature, "type", None)
            params = getattr(feature, "params", {})
        if isinstance(type_key, str):
            try:
                feature_type = FeatureType(type_key)
            except ValueError:
                feature_type = None
            token = FEATURE_TOKEN_MAP[feature_type] if feature_type is not None else None
            if token is None:
                token = _feature_token_from_name(type_key)
        elif isinstance(type_key, FeatureType):
            token = FEATURE_TOKEN_MAP[type_key]
        else:
            token = None
        if token is None:
            raise ValueError(f"no token for feature type {type_key!r}")
        tokens = [token]
        for value in params.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                _bin_idx, num_token = self.tokenizer.encode_length(float(value))
                tokens.append(num_token)
        return tokens

    # -- materials ------------------------------------------------------------

    def material_token(self, material_name: str) -> str:
        token = MATERIAL_TOKEN_MAP.get(material_name)
        if token is None:
            raise ValueError(f"no token for material {material_name!r}")
        return token

    # -- assemblies -----------------------------------------------------------

    def assembly_tokens(self, assembly: Any) -> list[str]:
        """Tokenize component names + mate types of an assembly."""
        tokens: list[str] = []
        if hasattr(assembly, "components"):
            for component in assembly.components:
                name = getattr(component, "name", None) or "COMPONENT"
                tokens.append(_sanitize_token(name))
        if hasattr(assembly, "constraints"):
            for constraint in assembly.constraints:
                mate = getattr(constraint, "mate_type", None) or getattr(constraint, "type", None)
                if mate is not None:
                    tokens.append(_sanitize_token(mate))
        return tokens

    # -- full sequence ---------------------------------------------------------

    def design_to_tokens(self, design: dict[str, Any]) -> list[str]:
        """Flatten a design dict into an ordered CAD token list.

        Recognised keys: ``primitives``, ``features``, ``material``,
        ``assembly``, ``gdt``, ``manufacturing``.
        """
        tokens: list[str] = []
        for primitive in design.get("primitives", []):
            tokens.extend(self.primitive_tokens(primitive))
        for feature in design.get("features", []):
            tokens.extend(self.feature_tokens(feature))
        material = design.get("material")
        if material:
            name = material if isinstance(material, str) else material.get("name", "")
            tokens.append(self.material_token(name))
        return tokens

    def to_sequence(
        self,
        design: dict[str, Any],
        text: str = "",
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Any:
        """Return a ``CADTokenSequence`` from an ``AutonomousCADTokenizer``."""
        tokens = self.design_to_tokens(design)
        if text:
            return self.tokenizer.encode_multimodal(text, tokens, add_bos, add_eos)
        return self.tokenizer.encode_cad_sequence(tokens, add_bos, add_eos)


def _primitive_kind_token(kind: str) -> str | None:
    mapping = {
        "BOX": "PRIM_BOX",
        "CYLINDER": "PRIM_CYLINDER",
        "SPHERE": "PRIM_SPHERE",
        "CONE": "PRIM_CONE",
        "TORUS": "PRIM_TORUS",
        "PRISM": "PRIM_PRISM",
        "PYRAMID": "PRIM_PYRAMID",
    }
    return mapping.get(kind)


def _feature_token_from_name(name: str) -> str | None:
    if not name:
        return None
    upper = str(name).upper()
    for token in ("EXTRUDE", "REVOLVE", "SWEEP", "LOFT", "FILLET", "CHAMFER", "SHELL", "HOLE"):
        if upper.startswith(token):
            return f"FEAT_{token}"
    if "UNION" in upper:
        return "FEAT_BOOL_UNION"
    if "CUT" in upper or "SUBTRACT" in upper:
        return "FEAT_BOOL_CUT"
    return None


def _sanitize_token(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "_" else "_" for c in name.upper())
    return f"ASM_{cleaned}" if not cleaned.startswith("ASM_") else cleaned


__all__ = [
    "FEATURE_TOKEN_MAP",
    "MATERIAL_TOKEN_MAP",
    "TokenizerBridge",
]
