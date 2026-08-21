"""All subsystem tests."""
import sys
sys.path.insert(0, 'src')


def test_basic_imports():
    """Test that all major modules can be imported."""
    import cadgenesis
    assert cadgenesis is not None


def test_adapters_import():
    """Test that adapters module can be imported."""
    from cadgenesis import adapters
    assert adapters is not None


def test_agents_import():
    """Test that agents module can be imported."""
    from cadgenesis import agents
    assert agents is not None