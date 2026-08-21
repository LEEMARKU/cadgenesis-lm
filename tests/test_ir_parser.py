import sys
sys.path.insert(0, 'src')

from cadgensis.ir.parser import IRParser


def test_ir_parser_init():
    parser = IRParser()
    assert parser is not None