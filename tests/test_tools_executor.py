import sys
sys.path.insert(0, 'src')

from cadgensis.tools.executor import ToolExecutor


def test_tool_executor_init():
    executor = ToolExecutor()
    assert executor is not None