# CAD Intelligence — API Reference

Concise usage reference for the main entry points of `cadgenesis.cad`.

## Geometry

```python
from cadgenesis.cad.geometry.core import Vec, Plane, Transform

a = Vec(1, 2, 3)
b = a + Vec(1, 0, 0)  # Vec(2, 2, 3)
n = a.norm()  # magnitude
p = Plane.xy()  # z = 0 plane
d = p.signed_distance(Vec(0, 0, 5))

t = Transform.translation(10, 0, 0)
rot = Transform.rotation(angle=90, axis=Vec(0, 0, 1))
world = t.composed(rot)
pt = world.apply(Vec(0, 0, 0))
```

Curves (Bezier + NURBS):

```python
from cadgenesis.cad.geometry.curves import bezier_curve, NurbsCurve, knot_vector

pts = bezier_curve([Vec(0, 0, 0), Vec(1, 2, 0), Vec(2, 0, 0)], samples=16)
curve = NurbsCurve(
    degree=2, control_points=[Vec(0, 0, 0), Vec(1, 1, 0), Vec(2, 0, 0)], knots=knot_vector(2, 3)
)
p = curve.evaluate(0.5)
```

## Parametric

```python
from cadgenesis.cad.parametric.sketch import Sketch, SketchProfile
from cadgenesis.cad.parametric.constraints import SketchConstraintSolver, GeometricConstraint

sk = Sketch()
sk.add_line(Vec(0, 0), Vec(10, 0), name="base")
sk.add_line(Vec(10, 0), Vec(10, 5), name="right")
sk.add_constraint(GeometricConstraint("DIMENSION", "base", value=20.0))
sol = SketchConstraintSolver().solve(sketch)  # ConstraintSolution(status, dof, residual)
```

Parameters:

```python
from cadgenesis.cad.parametric.parameters import Parameter, ParameterTable

tab = ParameterTable()
w = tab.declare("width", 10.0)
h = tab.add(Parameter("height", 5.0))
w.value = 12.0  # ExpressionParameter resolves dependencies
```

## Features

```python
from cadgenesis.cad.features.base import FeatureTree, Feature, FeatureType
from cadgenesis.cad.features.solids import Extrude

tree = FeatureTree()
tree.add(Extrude(name="base", sketch_ref="sketch1", params={"depth": 10.0}))
tree.add(Feature(feature_type=FeatureType.HOLE, name="h", params={"diameter": 6.0}))
sequence = tree.execution_order()
```

## Modeling (B-Rep + CSG)

```python
from cadgenesis.cad.modeling.brep import BRepSolid
from cadgenesis.cad.modeling.csg import CSGTree, CSGNode
from cadgenesis.cad.modeling.primitives import make_box, make_cylinder

solid = BRepSolid.from_prism(10, 5, 3)
problems = solid.validate()  # [] == valid

tree = CSGTree()
box = tree.new_leaf(make_box(10, 10, 10))
hole = tree.new_leaf(make_cylinder(2, 10))
cut = tree.new_binary("SUBTRACT", box, hole)
```

## Mesh

```python
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.mesh.io import read_stl, write_obj
from cadgenesis.cad.mesh.repair import diagnose, fill_holes
from cadgenesis.cad.mesh.simplify import quadric_simplify

mesh = Mesh.box(10, 5, 3)
volume = mesh.volume()  # divergence-theorem enclosed volume
write_obj(mesh, "part.obj")
read_mesh = read_stl("part.stl")  # OBJ/STL/PLY readers+writers
simple = quadric_simplify(mesh, 0.5)
```

## Assembly

```python
from cadgenesis.cad.assembly.assembly import Assembly
from cadgenesis.cad.assembly.mates import AssemblyConstraint, MateSolver, Reference

asm = Assembly("asm")
asm.add_part("plate", part_id="p1")
world = asm.world_transform("p1")
```

## Materials + GD&T

```python
from cadgenesis.cad.materials.database import MaterialDatabase

db = MaterialDatabase()
db.get("AISI 1045")  # default registry lookup
db["ABS"]  # alias lookup

from cadgenesis.cad.gdt import GDTSpecification, Datum, FeatureControlFrame, ManufacturingTolerance

spec = GDTSpecification(
    datums=[Datum(identifier="A")],
    control_frames=[
        FeatureControlFrame(characteristic="POSITION", tolerance=0.05, datums=[DatumReference("A")])
    ],
    manufacturing_tolerances=[
        ManufacturingTolerance(kind="LIMIT", lower_limit=9.9, upper_limit=10.1, feature="bore")
    ],
)
print(spec.validate())
```

## Manufacturing

```python
from cadgenesis.cad.manufacturing.process import ProcessSelector, ProcessSelection
from cadgenesis.cad.manufacturing.features import cnc_feature, print_feature

part = {
    "material_category": "metal",
    "batch_size": 100,
    "max_part_size_mm": 150.0,
    "required_group": "cnc",
}
best = ProcessSelector().select(part)  # ProcessSelection with score + reasons
best.best, best.by_group.get("cnc")
```

## Mechanisms

```python
from cadgenesis.cad.mechanisms.gears import SpurGear, GearPair
from cadgenesis.cad.mechanisms.cams import CamProfile
from cadgenesis.cad.mechanisms.joints import Joint, Mechanism
from cadgenesis.cad.mechanisms.linkages import FourBarLinkage

driver = SpurGear("d", module=2, teeth=20)
pair = GearPair(driver, SpurGear("e", module=2, teeth=40))
pair.ratio  # 2.0

cam = CamProfile(base_radius=20)
cam.add_rise_dwell_fall(rise=10, rise_span=120, dwell_span=60, fall_span=180)

fb = FourBarLinkage(ground=60, crank=20, coupler=70, rocker=40)
fb.is_grashof, fb.rocker_angle(45)
```

## Validation

```python
from cadgenesis.cad.validation.pipeline import CadValidator

report = CadValidator().validate(design_object)
report.passed  # all checks green
for r in report.results:
    print(r.name, r.passed, r.detail)
```

## Integration

```python
from cadgenesis.cad.integration.pipeline import CADIntelligencePipeline
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer

pipeline = CADIntelligencePipeline(tokenizer=AutonomousCADTokenizer.build_mini())
result = pipeline.run(
    {"material": "AISI 1045", "features": [{"type": "FILLET", "params": {"radius": 2.0}}]},
    name="test_block",
    text="make a steel block",
)
print(result.sequence.is_valid, result.memory_key, result.validation.passed)
```