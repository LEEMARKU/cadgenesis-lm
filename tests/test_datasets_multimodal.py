import sys
sys.path.insert(0, 'src')

from cadgensis.datasets.multimodal import MultimodalDataset


def test_multimodal_dataset_init():
    dataset = MultimodalDataset()
    assert dataset is not None