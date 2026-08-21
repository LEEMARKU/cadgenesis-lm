import sys
sys.path.insert(0, 'src')

from cadgenesis.agents.design._helpers import BoundaryCondition, LoadCase, Material, WorldObject


def test_boundary_condition_init():
    bc = BoundaryCondition(name='fixed', value=0.0)
    assert bc is not None


def test_load_case_init():
    lc = LoadCase(name='dead', magnitude=1.0)
    assert lc is not None


def test_material_init():
    mat = Material(name='steel', young_modulus=200e9)
    assert mat is not None


def test_world_object_init():
    obj = WorldObject(name='test_obj')
    assert obj is not None