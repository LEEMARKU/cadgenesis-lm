import sys
sys.path.insert(0, 'src')

from cadgensis.ir.schema import IRSchema


def test_ir_schema_init():
    schema = IRSchema()
    assert schema is not None