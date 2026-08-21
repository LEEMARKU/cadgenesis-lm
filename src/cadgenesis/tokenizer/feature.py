"""
cadgenesis.tokenizer.feature
==============================
CAD feature operation token family.

Purpose
-------
Defines tokens for all parametric CAD feature operations that transform
geometry.  These map directly to the operations available in parametric
modellers (CATIA, SolidWorks, FreeCAD, OpenCASCADE) and form the core
"instruction set" of the CAD generation language.

Token naming convention: FEAT_CATEGORY_OPERATION
e.g. FEAT_BASED_EXTRUDE, FEAT_DRESS_FILLET, FEAT_PATTERN_LINEAR

Architecture
------------
Feature tokens are grouped into operation categories:
1. Sketch-based features (add material)
2. Sketch-based features (remove material)
3. Dress-up operations
4. Boolean operations
5. Pattern operations
6. Transform operations
7. Surface operations
8. Sheet metal operations
9. Analysis / utility tokens
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Feature token lists — (token_str, description)
# ---------------------------------------------------------------------------

_BASED_ADDITIVE: list[tuple[str, str]] = [
    # Sketch → solid (additive)
    ("FEAT_EXTRUDE", "Extrude a closed sketch profile into a solid"),
    ("FEAT_EXTRUDE_BOSS", "Boss extrusion (adds material to existing solid)"),
    ("FEAT_REVOLVE", "Revolve a sketch profile around an axis"),
    ("FEAT_SWEEP", "Sweep a profile along a path curve"),
    ("FEAT_LOFT", "Loft between two or more profiles"),
    ("FEAT_RIB", "Thin-wall rib from open sketch profile"),
    ("FEAT_THICKEN", "Thicken a surface into a solid"),
    ("FEAT_FILL", "Fill a surface hole with a patch"),
    ("FEAT_EMBOSS", "Emboss a profile onto a surface"),
    ("FEAT_WRAP", "Wrap a sketch around a curved surface"),
]

_BASED_SUBTRACTIVE: list[tuple[str, str]] = [
    # Sketch → void (removes material)
    ("FEAT_CUT", "Extrude-cut (removes material from solid)"),
    ("FEAT_REVOLVE_CUT", "Revolve cut"),
    ("FEAT_SWEEP_CUT", "Sweep cut along a path"),
    ("FEAT_LOFT_CUT", "Loft cut between profiles"),
    ("FEAT_POCKET", "Pocket (blind or through extrude cut)"),
    ("FEAT_SLOT", "Slot cut from an open sketch"),
    ("FEAT_GROOVE", "Groove cut on a revolution surface"),
    ("FEAT_HOLE", "Hole feature (simple, counterbore, countersink)"),
    ("FEAT_HOLE_SIMPLE", "Simple through-hole or blind hole"),
    ("FEAT_HOLE_CB", "Counterbore hole"),
    ("FEAT_HOLE_CS", "Countersink hole"),
    ("FEAT_HOLE_TAPPED", "Tapped (threaded) hole"),
    ("FEAT_THREAD", "Cosmetic thread annotation"),
    ("FEAT_SHELL", "Shell operation — hollow out a solid"),
]

_DRESS_UP: list[tuple[str, str]] = [
    # Finishing / blending operations
    ("FEAT_FILLET", "Fillet edge(s) with a constant radius"),
    ("FEAT_FILLET_VAR", "Variable-radius fillet"),
    ("FEAT_FILLET_FACE", "Face fillet between two faces"),
    ("FEAT_CHAMFER", "Chamfer edge(s) (angle-distance or dist-dist)"),
    ("FEAT_DRAFT", "Draft angle applied to faces"),
    ("FEAT_TAPER", "Taper faces at an angle from parting line"),
    ("FEAT_DOME", "Dome feature (raised or depressed)"),
    ("FEAT_VENT", "Vent / louver feature"),
    ("FEAT_DEFORM", "Free-form deformation"),
]

_BOOLEAN: list[tuple[str, str]] = [
    ("FEAT_BOOL_UNION", "Boolean union of two or more solids"),
    ("FEAT_BOOL_CUT", "Boolean cut (subtract solid B from solid A)"),
    ("FEAT_BOOL_INTERSECT", "Boolean intersection of solids"),
    ("FEAT_BOOL_SPLIT", "Split a solid using a surface or plane"),
    ("FEAT_BOOL_SECTION", "Section / slice at a plane"),
]

_PATTERN: list[tuple[str, str]] = [
    ("FEAT_PATTERN_LIN", "Linear (rectangular) pattern of a feature"),
    ("FEAT_PATTERN_CIRC", "Circular pattern of a feature"),
    ("FEAT_PATTERN_CURVE", "Curve-driven pattern of a feature"),
    ("FEAT_PATTERN_FILL", "Fill pattern within a boundary"),
    ("FEAT_PATTERN_MIRROR", "Mirror feature across a plane"),
    ("FEAT_PATTERN_BODY", "Mirror entire body"),
]

_TRANSFORM: list[tuple[str, str]] = [
    ("FEAT_MOVE", "Translate body or face by a vector"),
    ("FEAT_ROTATE", "Rotate body or face around an axis"),
    ("FEAT_SCALE", "Scale body uniformly or non-uniformly"),
    ("FEAT_ALIGN", "Align body to a reference"),
    ("FEAT_OFFSET_BODY", "Offset all faces of a solid outward/inward"),
    ("FEAT_SPLIT_BODY", "Split solid into separate bodies"),
]

_SURFACE_OPS: list[tuple[str, str]] = [
    ("FEAT_SURF_EXTRUDE", "Extrude a curve/sketch into a surface"),
    ("FEAT_SURF_REVOLVE", "Revolve a curve into a surface"),
    ("FEAT_SURF_SWEEP", "Sweep a curve along a path"),
    ("FEAT_SURF_LOFT", "Loft between curves to form a surface"),
    ("FEAT_SURF_OFFSET", "Offset a surface at a constant distance"),
    ("FEAT_SURF_TRIM", "Trim a surface with another surface"),
    ("FEAT_SURF_UNTRIM", "Un-trim a surface to its natural bounds"),
    ("FEAT_SURF_KNIT", "Knit (join) multiple surfaces into one"),
    ("FEAT_SURF_THICKEN", "Thicken a surface into a solid"),
    ("FEAT_SURF_DELETE", "Delete a face from a solid (heals automatically)"),
]

_SHEET_METAL: list[tuple[str, str]] = [
    ("FEAT_SM_FLANGE", "Sheet metal flange"),
    ("FEAT_SM_BEND", "Sheet metal bend"),
    ("FEAT_SM_UNFOLD", "Unfold / flatten sheet metal part"),
    ("FEAT_SM_FOLD", "Fold flat sheet metal pattern"),
    ("FEAT_SM_JUNCTURE", "Sheet metal juncture / weld"),
    ("FEAT_SM_CORNER", "Sheet metal corner relief"),
    ("FEAT_SM_LOUVER", "Sheet metal louver"),
    ("FEAT_SM_FORM", "Sheet metal form tool application"),
]

_UTILITY: list[tuple[str, str]] = [
    # Utility / structural tokens used within feature sequences
    ("FEAT_BEGIN", "Begin of a feature definition block"),
    ("FEAT_END", "End of a feature definition block"),
    ("FEAT_PARAM_BEGIN", "Begin of feature parameter list"),
    ("FEAT_PARAM_END", "End of feature parameter list"),
    ("FEAT_CONDITION", "Feature termination condition (blind/through/to_face/…)"),
    ("FEAT_COND_BLIND", "Blind termination — depth specified by parameter"),
    ("FEAT_COND_THROUGH", "Through-all termination"),
    ("FEAT_COND_TO_FACE", "Terminate at a specified face"),
    ("FEAT_COND_TO_BODY", "Terminate at a specified body"),
    ("FEAT_COND_MIDPLANE", "Symmetric mid-plane termination"),
    ("FEAT_FLIP", "Flip the direction of a feature"),
    ("FEAT_REVERSE", "Reverse material addition/removal direction"),
    ("FEAT_DRAFT_IN", "Apply inward draft to a feature"),
    ("FEAT_DRAFT_OUT", "Apply outward draft to a feature"),
    ("FEAT_THIN", "Thin-feature modifier (wall thickness)"),
    ("FEAT_MERGE", "Merge result with existing body"),
    ("FEAT_NO_MERGE", "Do not merge (create new body)"),
]

# All feature tokens in registration order
_ALL_FEATURE_TOKENS: list[tuple[str, str]] = (
    _BASED_ADDITIVE
    + _BASED_SUBTRACTIVE
    + _DRESS_UP
    + _BOOLEAN
    + _PATTERN
    + _TRANSFORM
    + _SURFACE_OPS
    + _SHEET_METAL
    + _UTILITY
)


# ---------------------------------------------------------------------------
# FeatureTokenizer
# ---------------------------------------------------------------------------


class FeatureTokenizer:
    """
    Registers all CAD feature operation tokens into a CADVocabulary.

    Usage::

        from cadgenesis.tokenizer.vocabulary import CADVocabulary
        from cadgenesis.tokenizer.feature import FeatureTokenizer

        vocab = CADVocabulary()
        FeatureTokenizer.populate(vocab)
        print(vocab["FEAT_EXTRUDE"])   # → token id
    """

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        for token_str, desc in _ALL_FEATURE_TOKENS:
            vocab.register(token_str, TokenFamily.FEATURE, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_FEATURE_TOKENS]

    @classmethod
    def additive_tokens(cls) -> list[str]:
        return [t for t, _ in _BASED_ADDITIVE]

    @classmethod
    def subtractive_tokens(cls) -> list[str]:
        return [t for t, _ in _BASED_SUBTRACTIVE]

    @classmethod
    def dress_up_tokens(cls) -> list[str]:
        return [t for t, _ in _DRESS_UP]

    @classmethod
    def boolean_tokens(cls) -> list[str]:
        return [t for t, _ in _BOOLEAN]

    @classmethod
    def pattern_tokens(cls) -> list[str]:
        return [t for t, _ in _PATTERN]

    @classmethod
    def sheet_metal_tokens(cls) -> list[str]:
        return [t for t, _ in _SHEET_METAL]
