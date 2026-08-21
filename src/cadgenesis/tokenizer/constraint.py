"""
cadgenesis.tokenizer.constraint
================================
Parametric and geometric constraint token family.

Purpose
-------
Constraint tokens encode the relationships between geometric entities that
define a fully-constrained parametric CAD model.  These are the "rules" that
the geometry must satisfy, as distinct from the geometry itself.

Two constraint classes:
1. **Geometric constraints** — qualitative relationships
   (coincident, parallel, perpendicular, tangent, …)
2. **Dimensional constraints** — quantitative, with a numeric parameter token
   following the constraint token (e.g. CON_DIM_DIST NUM_050)

All constraint tokens share the CON_ prefix for easy family identification.

Architecture
------------
    ConstraintTokenizer
    ├── _GEOMETRIC_CONSTRAINTS   — qualitative topology constraints
    ├── _DIMENSIONAL_CONSTRAINTS — quantitative measurement constraints
    ├── _ASSEMBLY_CONSTRAINTS    — mating / alignment constraints
    ├── _ENGINEERING_CONSTRAINTS — physical / manufacturing constraints
    └── populate(vocab)
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Constraint token lists
# ---------------------------------------------------------------------------

_GEOMETRIC_CONSTRAINTS: list[tuple[str, str]] = [
    # Point / curve coincidence
    ("CON_COINCIDENT", "Entities share a common point"),
    ("CON_COLLINEAR", "Two lines share the same infinite line"),
    ("CON_COPLANAR", "Entities lie on the same plane"),
    # Direction relationships
    ("CON_PARALLEL", "Two lines / axes are parallel"),
    ("CON_PERPENDICULAR", "Two entities are perpendicular"),
    ("CON_HORIZONTAL", "Entity is horizontal (parallel to X-axis)"),
    ("CON_VERTICAL", "Entity is vertical (parallel to Y-axis)"),
    # Tangency / smoothness
    ("CON_TANGENT", "Two curves / surfaces are tangent at a point"),
    ("CON_CURVATURE", "G2 curvature-continuous tangency"),
    ("CON_SMOOTH", "G1 smooth (tangent-continuous) constraint"),
    # Symmetry
    ("CON_SYMMETRIC", "Entities are symmetric about a line / plane"),
    ("CON_MIDPOINT", "Point is at the midpoint of a line"),
    # Equality
    ("CON_EQUAL_LEN", "Two lines have equal length"),
    ("CON_EQUAL_RAD", "Two arcs / circles have equal radius"),
    ("CON_EQUAL_DIST", "Two measurements are equal"),
    # Fixed
    ("CON_FIXED", "Entity position is fixed in space"),
    ("CON_FIXED_DIR", "Direction of entity is fixed"),
    # Concentricity / concentric
    ("CON_CONCENTRIC", "Two arcs / circles share the same center"),
    # Intersection
    ("CON_INTERSECT", "Two curves intersect at a point"),
    # On-entity
    ("CON_ON_CURVE", "Point lies on a curve"),
    ("CON_ON_SURFACE", "Point / curve lies on a surface"),
    ("CON_ON_PLANE", "Entities lie on a given plane"),
    ("CON_ON_AXIS", "Entity lies on an axis"),
    ("CON_PIERCE", "Curve pierces a surface at a point"),
]

_DIMENSIONAL_CONSTRAINTS: list[tuple[str, str]] = [
    # Linear dimensions (followed by a NUM_xxx token)
    ("CON_DIM_LENGTH", "Linear length constraint"),
    ("CON_DIM_DIST", "Distance between two entities"),
    ("CON_DIM_DIST_X", "Horizontal distance between two points"),
    ("CON_DIM_DIST_Y", "Vertical distance between two points"),
    ("CON_DIM_DIST_Z", "Depth distance between two points"),
    # Angular dimensions
    ("CON_DIM_ANGLE", "Angle between two lines"),
    ("CON_DIM_ANGLE_3PT", "Angle defined by three points"),
    # Radial / diameter
    ("CON_DIM_RADIUS", "Radius of circle / arc"),
    ("CON_DIM_DIAMETER", "Diameter of circle / arc"),
    # Other
    ("CON_DIM_OFFSET", "Offset distance from a reference"),
    ("CON_DIM_CHORD", "Chord length of an arc"),
    ("CON_DIM_ARC_LEN", "Arc length"),
]

_ASSEMBLY_CONSTRAINTS: list[tuple[str, str]] = [
    ("CON_MATE", "Faces are mated (touching / flush)"),
    ("CON_ALIGN", "Faces / axes are aligned (same direction)"),
    ("CON_FLUSH", "Face surfaces are coplanar"),
    ("CON_GEAR", "Gear ratio constraint between two revolute joints"),
    ("CON_RACK_PINION", "Rack-and-pinion constraint"),
    ("CON_SCREW", "Screw / helical constraint"),
    ("CON_CAM", "Cam-follower contact constraint"),
    ("CON_SLOT_JOINT", "Slot joint (translation along a curve)"),
    ("CON_BALL_JOINT", "Ball-and-socket joint (3 rotational DOF)"),
    ("CON_PIVOT", "Pivot / pin joint (1 rotational DOF)"),
    ("CON_SLIDER", "Slider joint (1 translational DOF)"),
    ("CON_RIGID", "Rigid joint (0 DOF, fixed relationship)"),
    ("CON_SPRING", "Spring / compliant joint"),
    ("CON_WIRE", "Flexible wire / cable constraint"),
    ("CON_BELT", "Belt / chain drive constraint"),
]

_ENGINEERING_CONSTRAINTS: list[tuple[str, str]] = [
    # Manufacturability
    ("CON_MIN_WALL", "Minimum wall thickness constraint"),
    ("CON_MIN_RADIUS", "Minimum corner / fillet radius for tooling"),
    ("CON_MAX_DEPTH", "Maximum hole / pocket depth"),
    ("CON_DRAFT_REQ", "Required draft angle for moulding"),
    ("CON_SYMMETRY_MFG", "Manufacturing symmetry requirement"),
    # Structural
    ("CON_STRESS_MAX", "Maximum allowable stress"),
    ("CON_DEFL_MAX", "Maximum allowable deflection"),
    ("CON_MASS_MAX", "Maximum mass constraint"),
    ("CON_MASS_MIN", "Minimum mass constraint"),
    ("CON_FREQ_MIN", "Minimum natural frequency constraint"),
    # Tolerance
    ("CON_TOL_LINEAR", "Linear dimensional tolerance"),
    ("CON_TOL_ANGULAR", "Angular tolerance"),
    ("CON_TOL_FORM", "Form tolerance (flatness, roundness, …)"),
    ("CON_TOL_POSITION", "True-position tolerance (GD&T)"),
    ("CON_TOL_RUNOUT", "Runout tolerance (circular / total)"),
    # Clearance / fit
    ("CON_CLEARANCE", "Minimum clearance between parts"),
    ("CON_INTERFERENCE", "Interference fit specification"),
]

# Structural / utility tokens for constraint sequences
_CONSTRAINT_UTILITY: list[tuple[str, str]] = [
    ("CON_BEGIN", "Begin a constraint block"),
    ("CON_END", "End a constraint block"),
    ("CON_REF", "Reference entity for a constraint"),
    ("CON_TARGET", "Target entity for a constraint"),
    ("CON_UNDER", "Marks entity as under-constrained"),
    ("CON_FULLY", "Marks sketch / model as fully constrained"),
    ("CON_OVER", "Marks entity as over-constrained (error)"),
]

_ALL_CONSTRAINT_TOKENS: list[tuple[str, str]] = (
    _GEOMETRIC_CONSTRAINTS
    + _DIMENSIONAL_CONSTRAINTS
    + _ASSEMBLY_CONSTRAINTS
    + _ENGINEERING_CONSTRAINTS
    + _CONSTRAINT_UTILITY
)


# ---------------------------------------------------------------------------
# ConstraintTokenizer
# ---------------------------------------------------------------------------


class ConstraintTokenizer:
    """Registers all constraint tokens into a CADVocabulary."""

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        for token_str, desc in _ALL_CONSTRAINT_TOKENS:
            vocab.register(token_str, TokenFamily.CONSTRAINT, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_CONSTRAINT_TOKENS]

    @classmethod
    def geometric_tokens(cls) -> list[str]:
        return [t for t, _ in _GEOMETRIC_CONSTRAINTS]

    @classmethod
    def dimensional_tokens(cls) -> list[str]:
        return [t for t, _ in _DIMENSIONAL_CONSTRAINTS]

    @classmethod
    def engineering_tokens(cls) -> list[str]:
        return [t for t, _ in _ENGINEERING_CONSTRAINTS]
