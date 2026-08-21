import sys
sys.path.insert(0, 'src')

from cadgensis.ir.diff import DiffEngine


def test_diff_engine_init():
    engine = DiffEngine()
    assert engine is not None