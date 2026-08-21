import sys
sys.path.insert(0, 'src')

from cadgensis.cad.integration.execution_bridge import ExecutionBridge


def test_execution_bridge_init():
    bridge = ExecutionBridge()
    assert bridge is not None