import sys
sys.path.insert(0, 'src')

from cadgensis.execution.execution_engine import ExecutionEngine


def test_execution_engine_init():
    engine = ExecutionEngine()
    assert engine is not None