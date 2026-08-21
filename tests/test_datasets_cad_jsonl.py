import sys
sys.path.insert(0, 'src')

from cadgensis.datasets.cad_jsonl import CADJSONL


def test_cad_jsonl_init():
    dataset = CADJSONL()
    assert dataset is not None