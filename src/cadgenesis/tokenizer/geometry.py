"""
cadgenesis.tokenizer.geometry
==============================
Geometric primitive and B-Rep element token family.

Purpose
-------
Defines tokens for all geometric entities that a CAD model references:
- 0-D: point, vertex
- 1-D: line, arc, circle, spline, polyline, helix
- 2-D: plane, surface, face (planar, cylindrical, conical, toroidal, NURBS)
- 3-D: solid body, volume
- B-Rep: shell, wire, compound
- Coordinate system tokens: origin, x_axis, y_axis, z_axis
- Sketch plane tokens
- Reference geometry

Each token is a string in SCREAMING_SNAKE_CASE prefixed by the entity
category, e.g. PRIM_BOX, PRIM_CYLINDER, SURF_PLANE, BREP_EDGE, etc.

Architecture
------------
::

    GeometryTokenizer
    ├── _PRIMITIVE_TOKENS   — basic solid primitives
    ├── _CURVE_TOKENS       — 1-D curve entities
    ├── _SURFACE_TOKENS     — 2-D surface entities
    ├── _BREP_TOKENS        — B-Rep topology elements
    ├── _SKETCH_TOKENS      — 2-D sketch entities
    ├── _CSYS_TOKENS        — coordinate system references
    └── populate(vocab)     — registers all above into CADVocabulary

Token string naming convention:
    PREFIX_NAME
    where PREFIX is one of: PRIM, CURVE, SURF, BREP, SK, CSYS, REF
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Token lists — (token_str, description)
# ---------------------------------------------------------------------------

_PRIMITIVE_TOKENS: list[tuple[str, str]] = [
    # Solid primitives
    ("PRIM_BOX", "Rectangular box / cuboid primitive"),
    ("PRIM_CYLINDER", "Cylinder primitive"),
    ("PRIM_SPHERE", "Sphere primitive"),
    ("PRIM_CONE", "Cone primitive"),
    ("PRIM_TORUS", "Torus (donut) primitive"),
    ("PRIM_WEDGE", "Wedge primitive"),
    ("PRIM_PRISM", "Regular prism primitive"),
    ("PRIM_PYRAMID", "Regular pyramid primitive"),
    ("PRIM_ELLIPSOID", "Ellipsoid primitive"),
    ("PRIM_CAPSULE", "Capsule (cylinder + two hemispheres)"),
    # Derived / compound solids
    ("PRIM_SOLID", "Generic solid body (union/intersect result)"),
    ("PRIM_SHELL", "Open shell (non-closed solid)"),
    ("PRIM_COMPOUND", "Compound of multiple solids"),
]

_CURVE_TOKENS: list[tuple[str, str]] = [
    ("CURVE_LINE", "Straight line segment"),
    ("CURVE_ARC", "Circular arc (defined by center, radius, angles)"),
    ("CURVE_CIRCLE", "Full circle"),
    ("CURVE_ELLIPSE", "Ellipse"),
    ("CURVE_ELLIPSE_ARC", "Elliptical arc"),
    ("CURVE_BSPLINE", "B-spline curve"),
    ("CURVE_BEZIER", "Bézier curve"),
    ("CURVE_NURBS", "NURBS curve"),
    ("CURVE_POLYLINE", "Polyline (sequence of connected line segments)"),
    ("CURVE_HELIX", "Helix / spiral curve"),
    ("CURVE_INVOLUTE", "Involute curve (gear tooth profile)"),
    ("CURVE_PARABOLA", "Parabolic curve"),
    ("CURVE_HYPERBOLA", "Hyperbolic curve"),
    ("CURVE_OFFSET", "Offset curve (equidistant from a base curve)"),
    ("CURVE_TRIM", "Trimmed portion of a curve"),
    ("CURVE_INTERSECT", "Intersection curve of two surfaces"),
]

_SURFACE_TOKENS: list[tuple[str, str]] = [
    ("SURF_PLANE", "Infinite or bounded plane"),
    ("SURF_CYLINDER", "Cylindrical surface"),
    ("SURF_CONE", "Conical surface"),
    ("SURF_SPHERE_S", "Spherical surface"),
    ("SURF_TORUS_S", "Toroidal surface"),
    ("SURF_NURBS", "NURBS surface (arbitrary free-form)"),
    ("SURF_BSPLINE", "B-spline surface"),
    ("SURF_BEZIER", "Bézier patch"),
    ("SURF_RULED", "Ruled surface (linear interpolation between curves)"),
    ("SURF_REVOL", "Surface of revolution"),
    ("SURF_EXTRUDE", "Extruded surface (swept along a direction)"),
    ("SURF_OFFSET", "Offset surface"),
    ("SURF_FILLET", "Fillet surface"),
    ("SURF_TRIMMED", "Trimmed surface (face)"),
]

_BREP_TOKENS: list[tuple[str, str]] = [
    ("BREP_VERTEX", "B-Rep vertex (0-D topological entity)"),
    ("BREP_EDGE", "B-Rep edge (1-D topological entity)"),
    ("BREP_WIRE", "B-Rep wire (closed loop of edges)"),
    ("BREP_FACE", "B-Rep face (bounded 2-D surface region)"),
    ("BREP_SHELL", "B-Rep shell (connected set of faces)"),
    ("BREP_SOLID", "B-Rep solid (closed shell enclosing volume)"),
    ("BREP_COMPOUND", "B-Rep compound (collection of shapes)"),
    # Orientation
    ("BREP_FWD", "Forward orientation of B-Rep entity"),
    ("BREP_REV", "Reversed orientation of B-Rep entity"),
    # Adjacency
    ("BREP_COEDGE", "Co-edge (edge use within a loop)"),
    ("BREP_LOOP", "Loop (ordered sequence of co-edges bounding a face)"),
]

_SKETCH_TOKENS: list[tuple[str, str]] = [
    ("SK_POINT", "Sketch point"),
    ("SK_LINE", "Sketch line segment"),
    ("SK_ARC", "Sketch arc"),
    ("SK_CIRCLE", "Sketch full circle"),
    ("SK_RECT", "Sketch rectangle (axis-aligned)"),
    ("SK_POLY", "Sketch polygon (n-gon)"),
    ("SK_SLOT", "Sketch slot (line + two arcs)"),
    ("SK_ELLIPSE", "Sketch ellipse"),
    ("SK_SPLINE", "Sketch spline through control points"),
    ("SK_CONSTRUCTION", "Construction geometry (reference, not extruded)"),
    ("SK_MIRROR_AXIS", "Mirror axis line in sketch"),
    ("SK_OFFSET", "Offset sketch entity"),
    ("SK_TRIM", "Trimmed sketch segment"),
    ("SK_EXTEND", "Extended sketch segment"),
    ("SK_FILLET_2D", "2-D fillet between two sketch entities"),
    ("SK_CHAMFER_2D", "2-D chamfer between two sketch entities"),
    ("SK_CLOSE", "Close sketch profile (explicit closure token)"),
    ("SK_OPEN", "Open sketch profile (explicit open token)"),
]

_CSYS_TOKENS: list[tuple[str, str]] = [
    ("CSYS_WORLD", "World / global coordinate system"),
    ("CSYS_LOCAL", "Local coordinate system"),
    ("CSYS_FACE", "Coordinate system derived from a face"),
    ("CSYS_EDGE", "Coordinate system derived from an edge"),
    ("CSYS_VERTEX", "Coordinate system at a vertex"),
    ("CSYS_ORIGIN", "Origin point of the active coordinate system"),
    ("CSYS_X", "X-axis direction vector"),
    ("CSYS_Y", "Y-axis direction vector"),
    ("CSYS_Z", "Z-axis direction vector"),
    ("CSYS_NORMAL", "Surface normal direction"),
    ("CSYS_TANGENT", "Curve tangent direction"),
]

_REF_TOKENS: list[tuple[str, str]] = [
    ("REF_PLANE_XY", "Reference plane: XY (top)"),
    ("REF_PLANE_XZ", "Reference plane: XZ (front)"),
    ("REF_PLANE_YZ", "Reference plane: YZ (right)"),
    ("REF_AXIS_X", "Reference axis: X"),
    ("REF_AXIS_Y", "Reference axis: Y"),
    ("REF_AXIS_Z", "Reference axis: Z"),
    ("REF_POINT", "Reference point"),
    ("REF_PLANE_OFFSET", "Offset reference plane"),
    ("REF_PLANE_ANGLED", "Angled reference plane"),
    ("REF_PLANE_TANGENT", "Tangent reference plane"),
    ("REF_AXIS_EDGE", "Reference axis from an edge"),
    ("REF_AXIS_2PT", "Reference axis from two points"),
]

# All geometry token groups
_ALL_GEOMETRY_TOKENS: list[tuple[str, str]] = (
    _PRIMITIVE_TOKENS
    + _CURVE_TOKENS
    + _SURFACE_TOKENS
    + _BREP_TOKENS
    + _SKETCH_TOKENS
    + _CSYS_TOKENS
    + _REF_TOKENS
)


# ---------------------------------------------------------------------------
# GeometryTokenizer
# ---------------------------------------------------------------------------


class GeometryTokenizer:
    """
    Registers all geometry tokens into a CADVocabulary.

    Usage::

        from cadgenesis.tokenizer.vocabulary import CADVocabulary
        from cadgenesis.tokenizer.geometry import GeometryTokenizer

        vocab = CADVocabulary()
        GeometryTokenizer.populate(vocab)
        print(vocab["PRIM_BOX"])   # → token id
    """

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        """Register all geometry tokens into the given vocabulary."""
        for token_str, desc in _ALL_GEOMETRY_TOKENS:
            vocab.register(token_str, TokenFamily.GEOMETRY, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_GEOMETRY_TOKENS]

    @classmethod
    def primitive_tokens(cls) -> list[str]:
        return [t for t, _ in _PRIMITIVE_TOKENS]

    @classmethod
    def sketch_tokens(cls) -> list[str]:
        return [t for t, _ in _SKETCH_TOKENS]

    @classmethod
    def brep_tokens(cls) -> list[str]:
        return [t for t, _ in _BREP_TOKENS]
