"""Test CAD materials database module."""
import sys
sys.path.insert(0, 'src')


def test_cad_materials_database():
    from cadgensis.cad.materials.database import MaterialsDatabase
    db = MaterialsDatabase()
    assert db is not None