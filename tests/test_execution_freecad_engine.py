import sys
sys.path.insert(0, 'src')

from cadgensis.execution.freecad_engine import FreeCADEngine


def test_freecad_engine_init():
    engine = FreeCADEngine()
    assert engine is not None