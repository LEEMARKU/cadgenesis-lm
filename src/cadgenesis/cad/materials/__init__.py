"""cadgenesis.cad.materials
=========================
Material intelligence: engineering material database and physical property
access.
"""

from cadgenesis.cad.materials.database import (
    MATERIAL_CATEGORIES,
    MATERIALS,
    Material,
    MaterialDatabase,
    MaterialProperty,
)

__all__ = [
    "MATERIALS",
    "MATERIAL_CATEGORIES",
    "Material",
    "MaterialDatabase",
    "MaterialProperty",
]
