import sys
sys.path.insert(0, 'src')

from cadgensis.tools.schema import ToolSchema


def test_tool_schema_init():
    schema = ToolSchema()
    assert schema is not None