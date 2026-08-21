import sys
sys.path.insert(0, 'src')

from cadgensis.platform.dashboard import Dashboard


def test_dashboard_init():
    dashboard = Dashboard()
    assert dashboard is not None