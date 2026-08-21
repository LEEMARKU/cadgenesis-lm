import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.decoder import CADDecoder


def test_cad_decoder_init():
    decoder = CADDecoder()
    assert decoder is not None