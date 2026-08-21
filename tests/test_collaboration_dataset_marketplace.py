import sys
sys.path.insert(0, 'src')

from cadgensis.collaboration.dataset_marketplace import DatasetMarketplace


def test_dataset_marketplace_init():
    marketplace = DatasetMarketplace()
    assert marketplace is not None