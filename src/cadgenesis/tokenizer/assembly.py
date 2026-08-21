"""
cadgenesis.tokenizer.assembly
==============================
Assembly relationship token family.

Purpose
-------
Assembly tokens encode the structural relationships between components
in a multi-body CAD assembly.  They let the model represent how parts
are connected, mated, and organized into sub-assemblies.

Token categories:
1. Assembly structure (sub-assembly, component, instance)
2. Mating / positioning relationships
3. Kinematic joint types
4. Motion / degree-of-freedom declarations
5. BOM (Bill-of-Materials) tokens
6. Utility / structural tokens

Every assembly sequence is wrapped in <assembly_start> ... <assembly_end>
special tokens (defined in vocabulary.py) and uses ASSM_ prefix.
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Assembly token lists
# ---------------------------------------------------------------------------

_STRUCTURE_TOKENS: list[tuple[str, str]] = [
    ("ASSM_BEGIN", "Begin assembly definition"),
    ("ASSM_END", "End assembly definition"),
    ("ASSM_SUB", "Begin sub-assembly reference"),
    ("ASSM_PART", "Begin part (leaf component) reference"),
    ("ASSM_INSTANCE", "Create a new instance of an existing part/sub-assy"),
    ("ASSM_PATTERN_LIN", "Linear pattern of instances in assembly"),
    ("ASSM_PATTERN_CIRC", "Circular pattern of instances in assembly"),
    ("ASSM_EXPLODE", "Begin exploded view definition"),
    ("ASSM_EXPLODE_END", "End exploded view definition"),
    ("ASSM_FLEXIBLE", "Sub-assembly is flexible (each instance independent)"),
    ("ASSM_RIGID", "Sub-assembly is rigid (single DOF set)"),
]

_MATE_TOKENS: list[tuple[str, str]] = [
    # Standard mates
    ("ASSM_MATE_COINCIDENT", "Coincident mate (faces / edges / points touch)"),
    ("ASSM_MATE_PARALLEL", "Parallel mate (face normals are parallel)"),
    ("ASSM_MATE_PERP", "Perpendicular mate (face normals are ⊥)"),
    ("ASSM_MATE_TANGENT", "Tangent mate (face is tangent to curved surface)"),
    ("ASSM_MATE_CONCENTRIC", "Concentric mate (axes are coaxial)"),
    ("ASSM_MATE_DISTANCE", "Distance mate (offset by NUM_xxx distance)"),
    ("ASSM_MATE_ANGLE", "Angle mate (faces at ANG_xxx angle)"),
    ("ASSM_MATE_LOCK", "Locked (zero DOF) mate"),
    ("ASSM_MATE_WIDTH", "Width mate (centre between two faces)"),
    ("ASSM_MATE_PROFILE", "Profile-centre mate (radially centred)"),
    # Advanced mates
    ("ASSM_MATE_GEAR", "Gear mate (rotation ratio between components)"),
    ("ASSM_MATE_RACK", "Rack-and-pinion mate"),
    ("ASSM_MATE_SCREW", "Screw / helical mate (coupled linear+rotation)"),
    ("ASSM_MATE_UNIVERSAL", "Universal joint mate"),
    ("ASSM_MATE_HINGE", "Hinge mate (1 rotational DOF with limits)"),
    ("ASSM_MATE_SLIDER", "Slider / linear mate (1 translational DOF)"),
    ("ASSM_MATE_BELT", "Belt / chain mate (coupled linear velocities)"),
    ("ASSM_MATE_CAM", "Cam-follower mate"),
    ("ASSM_MATE_PATH", "Path mate (point follows curve)"),
    ("ASSM_MATE_LIMIT_LIN", "Linear limit mate (min/max distance)"),
    ("ASSM_MATE_LIMIT_ANG", "Angular limit mate (min/max angle)"),
]

_KINEMATIC_TOKENS: list[tuple[str, str]] = [
    ("ASSM_DOF", "Degrees-of-freedom declaration"),
    ("ASSM_DOF_TX", "Translational DOF: X-axis"),
    ("ASSM_DOF_TY", "Translational DOF: Y-axis"),
    ("ASSM_DOF_TZ", "Translational DOF: Z-axis"),
    ("ASSM_DOF_RX", "Rotational DOF: X-axis"),
    ("ASSM_DOF_RY", "Rotational DOF: Y-axis"),
    ("ASSM_DOF_RZ", "Rotational DOF: Z-axis"),
    ("ASSM_FIXED_IN_SPACE", "Component is fixed (ground)"),
    ("ASSM_DRIVEN", "Component is driven by another component"),
    ("ASSM_DRIVING", "Component drives another component"),
]

_INTERFACE_TOKENS: list[tuple[str, str]] = [
    ("ASSM_FACE_REF", "Reference face for a mate"),
    ("ASSM_EDGE_REF", "Reference edge for a mate"),
    ("ASSM_POINT_REF", "Reference point for a mate"),
    ("ASSM_AXIS_REF", "Reference axis for a mate"),
    ("ASSM_PLANE_REF", "Reference plane for a mate"),
    ("ASSM_MATE_ALIGN", "Mate alignment: same direction"),
    ("ASSM_MATE_ANTI", "Mate alignment: opposite direction"),
]

_BOM_TOKENS: list[tuple[str, str]] = [
    ("ASSM_BOM_QTY", "BOM quantity for this component [→ NUM_xxx]"),
    ("ASSM_BOM_PN", "Part number token (followed by text encoding)"),
    ("ASSM_BOM_DESC", "Part description (followed by text encoding)"),
    ("ASSM_BOM_VENDOR", "Vendor / supplier designation"),
    ("ASSM_BOM_STANDARD", "Standard / off-the-shelf component"),
    ("ASSM_BOM_CUSTOM", "Custom / made-to-order component"),
    ("ASSM_BOM_FASTENER", "Fastener (bolt, screw, nut, …)"),
    ("ASSM_BOM_BEARING", "Bearing component"),
    ("ASSM_BOM_SEAL", "Seal / gasket / O-ring component"),
    ("ASSM_BOM_SPRING", "Spring component"),
]

_ALL_ASSEMBLY_TOKENS: list[tuple[str, str]] = (
    _STRUCTURE_TOKENS + _MATE_TOKENS + _KINEMATIC_TOKENS + _INTERFACE_TOKENS + _BOM_TOKENS
)


# ---------------------------------------------------------------------------
# AssemblyTokenizer
# ---------------------------------------------------------------------------


class AssemblyTokenizer:
    """Registers all assembly tokens into a CADVocabulary."""

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        for token_str, desc in _ALL_ASSEMBLY_TOKENS:
            vocab.register(token_str, TokenFamily.ASSEMBLY, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_ASSEMBLY_TOKENS]

    @classmethod
    def mate_tokens(cls) -> list[str]:
        return [t for t, _ in _MATE_TOKENS]

    @classmethod
    def kinematic_tokens(cls) -> list[str]:
        return [t for t, _ in _KINEMATIC_TOKENS]

    @classmethod
    def bom_tokens(cls) -> list[str]:
        return [t for t, _ in _BOM_TOKENS]
