import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.transformer_block import CADTransformerBlock


def test_transformer_block_init():
    block = CADTransformerBlock()
    assert block is not None