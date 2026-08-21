"""Test world model modules."""
import sys
sys.path.insert(0, 'src')


def test_world_model_imports():
    from cadgensis import world_model
    assert world_model is not None


def test_affordances():
    from cadgensis.world_model.affordances import Affordances
    a = Affordances()
    assert a is not None


def test_design_intent():
    from cadgensis.world_model.design_intent import DesignIntent
    d = DesignIntent()
    assert d is not None