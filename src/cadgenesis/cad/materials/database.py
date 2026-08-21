"""cadgenesis.cad.materials.database
=================================
Material intelligence: a curated engineering material database with density,
elasticity and thermal properties, organised into the four engineering
families (metals, plastics, composites, ceramics).

Values are representative engineering nominal data (SI base units where
applicable) used for mass estimation, process selection and cost modelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MATERIAL_CATEGORIES = ("metal", "plastic", "composite", "ceramic", "other")


@dataclass
class MaterialProperty:
    """A named scalar material property with units."""

    name: str
    value: float
    units: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "units": self.units}


@dataclass
class Material:
    """An engineering material with physical properties."""

    name: str
    category: str
    properties: dict[str, float] = field(default_factory=dict)
    cost_per_kg_usd: float = 0.0
    aliases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in MATERIAL_CATEGORIES:
            raise ValueError(
                f"invalid material category {self.category!r}; "
                f"expected one of {MATERIAL_CATEGORIES}"
            )

    # -- property access ------------------------------------------------------
    def __getitem__(self, key: str) -> float:
        if key not in self.properties:
            raise KeyError(f"material {self.name!r} has no property {key!r}")
        return self.properties[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.properties.get(key, default)

    def density(self) -> float:
        """Density in kg/m^3."""
        return self.properties.get("density", 0.0)

    def youngs_modulus(self) -> float:
        """Young's modulus in GPa."""
        return self.properties.get("youngs_modulus_gpa", 0.0)

    def yield_strength(self) -> float:
        """Yield strength in MPa."""
        return self.properties.get("yield_strength_mpa", 0.0)

    def thermal_conductivity(self) -> float:
        """Thermal conductivity in W/(m.K)."""
        return self.properties.get("thermal_conductivity_w_mk", 0.0)

    def poisson_ratio(self) -> float:
        return self.properties.get("poisson_ratio", 0.3)

    def density_kg_per_cm3(self) -> float:
        return self.density() / 1e6

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "properties": dict(self.properties),
            "cost_per_kg_usd": self.cost_per_kg_usd,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Material:
        return cls(
            name=str(data["name"]),
            category=str(data["category"]),
            properties={str(k): float(v) for k, v in data.get("properties", {}).items()},
            cost_per_kg_usd=float(data.get("cost_per_kg_usd", 0.0)),
            aliases=[str(a) for a in data.get("aliases", [])],
        )


def _metal(
    name: str,
    density_gcc: float,
    youngs_gpa: float,
    yield_mpa: float,
    thermal_wmk: float,
    cost: float,
    aliases: list[str],
    **extra: float,
) -> Material:
    return Material(
        name,
        "metal",
        {
            "density": density_gcc * 1e3,
            "youngs_modulus_gpa": youngs_gpa,
            "yield_strength_mpa": yield_mpa,
            "thermal_conductivity_w_mk": thermal_wmk,
            "poisson_ratio": 0.33,
            **extra,
        },
        cost_per_kg_usd=cost,
        aliases=aliases,
    )


def _plastic(
    name: str,
    density_gcc: float,
    youngs_gpa: float,
    yield_mpa: float,
    thermal_wmk: float,
    cost: float,
    aliases: list[str],
    **extra: float,
) -> Material:
    return Material(
        name,
        "plastic",
        {
            "density": density_gcc * 1e3,
            "youngs_modulus_gpa": youngs_gpa,
            "yield_strength_mpa": yield_mpa,
            "thermal_conductivity_w_mk": thermal_wmk,
            "poisson_ratio": 0.4,
            **extra,
        },
        cost_per_kg_usd=cost,
        aliases=aliases,
    )


# ---------------------------------------------------------------------------
# Built-in materials
# ---------------------------------------------------------------------------

MATERIALS: dict[str, Material] = {
    # --- metals ----------------------------------------------------------------
    "AISI 1018": _metal("AISI 1018", 7.87, 205, 370, 51.9, 1.5, ["mild steel", "low carbon steel"]),
    "AISI 1045": _metal("AISI 1045", 7.85, 205, 450, 51.0, 2.0, ["medium carbon steel"]),
    "AISI 4140": _metal("AISI 4140", 7.85, 210, 655, 42.6, 2.5, ["alloy steel 4140"]),
    "D2 tool steel": _metal("D2 tool steel", 7.70, 210, 1800, 20.0, 8.0, ["tool steel D2"]),
    "H13": _metal("H13", 7.80, 215, 1450, 28.4, 9.0, ["tool steel H13"]),
    "AISI 52100": _metal("AISI 52100", 7.81, 210, 2034, 46.6, 5.0, ["bearing steel 52100"]),
    "304 stainless": _metal(
        "304 stainless", 8.00, 193, 215, 16.2, 3.0, ["SS304", "stainless steel 304"]
    ),
    "316 stainless": _metal(
        "316 stainless", 8.00, 193, 290, 16.3, 4.0, ["SS316", "stainless steel 316"]
    ),
    "17-4 PH": _metal("17-4 PH", 7.80, 196, 1170, 18.3, 6.0, ["stainless 17-4"]),
    "Al 6061-T6": _metal("Al 6061-T6", 2.70, 68.9, 276, 167, 3.0, ["aluminium 6061", "6061"]),
    "Al 7075-T6": _metal("Al 7075-T6", 2.81, 71.7, 503, 130, 5.0, ["aluminium 7075", "7075"]),
    "Al 2024-T3": _metal("Al 2024-T3", 2.78, 73.1, 345, 121, 5.5, ["aluminium 2024"]),
    "Al 5052": _metal("Al 5052", 2.68, 70.3, 193, 138, 4.0, ["aluminium 5052"]),
    "Ti-6Al-4V": _metal("Ti-6Al-4V", 4.43, 113.8, 880, 6.7, 40.0, ["titanium grade 5", "TC4"]),
    "Grade 2 Ti": _metal("Grade 2 Ti", 4.51, 105, 345, 16.0, 30.0, ["commercial pure titanium"]),
    "Copper C110": _metal("Copper C110", 8.94, 117, 70, 385, 9.0, ["copper"]),
    "Brass C360": _metal("Brass C360", 8.50, 97, 345, 115, 8.0, ["brass"]),
    "Bronze C932": _metal("Bronze C932", 8.70, 100, 220, 59, 10.0, ["bronze"]),
    "Beryllium Copper": _metal("Beryllium Copper", 8.25, 131, 1100, 105, 30.0, ["BeCu"]),
    "Magnesium AZ31": _metal("Magnesium AZ31", 1.77, 45, 200, 72, 6.0, ["magnesium"]),
    "Zamak 3": _metal("Zamak 3", 6.60, 96, 228, 113, 4.0, ["zinc die cast"]),
    "Inconel 718": _metal("Inconel 718", 8.19, 200, 1034, 11.4, 60.0, ["nickel superalloy"]),
    "Cast iron": _metal("Cast iron", 7.20, 110, 275, 55, 3.0, ["grey cast iron"]),
    "6061 Al": _metal("6061 Al", 2.70, 68.9, 276, 167, 3.0, ["al 6061"]),
    # --- plastics ----------------------------------------------------------------
    "ABS": _plastic("ABS", 1.04, 2.3, 43, 0.19, 2.5, ["acrylonitrile butadiene styrene"]),
    "PLA": _plastic("PLA", 1.24, 3.5, 60, 0.13, 3.0, ["polylactic acid"]),
    "PETG": _plastic("PETG", 1.27, 2.1, 50, 0.19, 3.5, []),
    "Nylon PA6": _plastic("Nylon PA6", 1.14, 3.0, 80, 0.25, 4.0, ["nylon 6", "PA6"]),
    "Nylon PA66": _plastic("Nylon PA66", 1.15, 3.1, 83, 0.25, 4.5, ["nylon 66"]),
    "POM": _plastic("POM", 1.42, 2.8, 71, 0.30, 4.0, ["acetal", "delrin"]),
    "PEEK": _plastic("PEEK", 1.32, 3.6, 97, 0.25, 80.0, ["polyether ether ketone"]),
    "Polycarbonate": _plastic("Polycarbonate", 1.20, 2.4, 63, 0.20, 4.5, ["PC"]),
    "Polypropylene": _plastic("Polypropylene", 0.91, 1.5, 35, 0.17, 2.0, ["PP"]),
    "HDPE": _plastic("HDPE", 0.95, 0.8, 30, 0.49, 2.0, ["high density polyethylene"]),
    "LDPE": _plastic("LDPE", 0.92, 0.3, 12, 0.33, 2.0, ["low density polyethylene"]),
    "PVC": _plastic("PVC", 1.40, 3.2, 55, 0.17, 2.5, []),
    "PTFE": _plastic("PTFE", 2.20, 0.5, 25, 0.25, 20.0, ["teflon"]),
    "Ultem": _plastic("Ultem", 1.27, 3.5, 105, 0.22, 70.0, ["PEI"]),
    "Epoxy": _plastic("Epoxy", 1.16, 3.0, 75, 0.25, 10.0, []),
    "Polyurethane": _plastic("Polyurethane", 1.20, 0.03, 25, 0.20, 8.0, ["PU"]),
    "Silicone": _plastic("Silicone", 1.10, 0.002, 3.0, 0.20, 15.0, []),
    "Nitrile": _plastic("Nitrile", 1.00, 0.01, 8.0, 0.25, 8.0, ["NBR"]),
    "EPDM": _plastic("EPDM", 0.95, 0.008, 6.0, 0.25, 6.0, []),
    # --- composites --------------------------------------------------------------
    "CFRP": Material(
        "CFRP",
        "composite",
        {
            "density": 1600.0,
            "youngs_modulus_gpa": 135.0,
            "yield_strength_mpa": 1500.0,
            "thermal_conductivity_w_mk": 7.0,
            "poisson_ratio": 0.3,
        },
        cost_per_kg_usd=40.0,
        aliases=["carbon fiber composite"],
    ),
    "GFRP": Material(
        "GFRP",
        "composite",
        {
            "density": 1900.0,
            "youngs_modulus_gpa": 40.0,
            "yield_strength_mpa": 400.0,
            "thermal_conductivity_w_mk": 0.4,
            "poisson_ratio": 0.3,
        },
        cost_per_kg_usd=8.0,
        aliases=["fiberglass"],
    ),
    "Aramid": Material(
        "Aramid",
        "composite",
        {
            "density": 1450.0,
            "youngs_modulus_gpa": 90.0,
            "yield_strength_mpa": 3000.0,
            "thermal_conductivity_w_mk": 0.04,
            "poisson_ratio": 0.35,
        },
        cost_per_kg_usd=35.0,
        aliases=["kevlar"],
    ),
    "Basalt": Material(
        "Basalt",
        "composite",
        {
            "density": 2600.0,
            "youngs_modulus_gpa": 85.0,
            "yield_strength_mpa": 1000.0,
            "thermal_conductivity_w_mk": 0.03,
            "poisson_ratio": 0.2,
        },
        cost_per_kg_usd=10.0,
        aliases=["basalt fiber"],
    ),
    "Al-SiC MMC": Material(
        "Al-SiC MMC",
        "composite",
        {
            "density": 2900.0,
            "youngs_modulus_gpa": 180.0,
            "yield_strength_mpa": 400.0,
            "thermal_conductivity_w_mk": 200.0,
            "poisson_ratio": 0.3,
        },
        cost_per_kg_usd=60.0,
        aliases=["metal matrix composite"],
    ),
    "Plywood": Material(
        "Plywood",
        "composite",
        {
            "density": 600.0,
            "youngs_modulus_gpa": 10.0,
            "yield_strength_mpa": 30.0,
            "thermal_conductivity_w_mk": 0.13,
            "poisson_ratio": 0.3,
        },
        cost_per_kg_usd=1.5,
        aliases=["wood"],
    ),
    "Concrete": Material(
        "Concrete",
        "composite",
        {
            "density": 2400.0,
            "youngs_modulus_gpa": 30.0,
            "yield_strength_mpa": 30.0,
            "thermal_conductivity_w_mk": 1.7,
            "poisson_ratio": 0.2,
        },
        cost_per_kg_usd=0.1,
        aliases=[],
    ),
    # --- ceramics -----------------------------------------------------------------
    "Alumina": Material(
        "Alumina",
        "ceramic",
        {
            "density": 3960.0,
            "youngs_modulus_gpa": 370.0,
            "yield_strength_mpa": 300.0,
            "thermal_conductivity_w_mk": 30.0,
            "poisson_ratio": 0.22,
        },
        cost_per_kg_usd=25.0,
        aliases=["Al2O3"],
    ),
    "Zirconia": Material(
        "Zirconia",
        "ceramic",
        {
            "density": 6050.0,
            "youngs_modulus_gpa": 210.0,
            "yield_strength_mpa": 900.0,
            "thermal_conductivity_w_mk": 3.0,
            "poisson_ratio": 0.31,
        },
        cost_per_kg_usd=60.0,
        aliases=["ZrO2"],
    ),
    "Silicon carbide": Material(
        "Silicon carbide",
        "ceramic",
        {
            "density": 3210.0,
            "youngs_modulus_gpa": 410.0,
            "yield_strength_mpa": 500.0,
            "thermal_conductivity_w_mk": 120.0,
            "poisson_ratio": 0.14,
        },
        cost_per_kg_usd=50.0,
        aliases=["SiC"],
    ),
    "Silicon nitride": Material(
        "Silicon nitride",
        "ceramic",
        {
            "density": 3290.0,
            "youngs_modulus_gpa": 310.0,
            "yield_strength_mpa": 600.0,
            "thermal_conductivity_w_mk": 30.0,
            "poisson_ratio": 0.27,
        },
        cost_per_kg_usd=55.0,
        aliases=["Si3N4"],
    ),
    "Borosilicate glass": Material(
        "Borosilicate glass",
        "ceramic",
        {
            "density": 2230.0,
            "youngs_modulus_gpa": 64.0,
            "yield_strength_mpa": 90.0,
            "thermal_conductivity_w_mk": 1.2,
            "poisson_ratio": 0.2,
        },
        cost_per_kg_usd=3.0,
        aliases=["pyrex"],
    ),
}


class MaterialDatabase:
    """Registry of engineering materials with lookup by name or alias."""

    def __init__(self, materials: dict[str, Material] | None = None) -> None:
        self._materials: dict[str, Material] = dict(materials or MATERIALS)
        self._alias_map: dict[str, str] = {}
        for name, material in self._materials.items():
            self._alias_map[name.lower()] = name
            for alias in material.aliases:
                self._alias_map[alias.lower()] = name

    def __contains__(self, name: object) -> bool:
        return str(name).lower() in self._alias_map

    def __getitem__(self, name: str) -> Material:
        return self.get(name)

    def get(self, name: str) -> Material:
        key = self._alias_map.get(name.strip().lower())
        if key is None:
            raise KeyError(f"unknown material {name!r}")
        return self._materials[key]

    def names(self) -> list[str]:
        return list(self._materials)

    def by_category(self, category: str) -> list[Material]:
        return [m for m in self._materials.values() if m.category == category]

    def metals(self) -> list[Material]:
        return self.by_category("metal")

    def plastics(self) -> list[Material]:
        return self.by_category("plastic")

    def composites(self) -> list[Material]:
        return self.by_category("composite")

    def ceramics(self) -> list[Material]:
        return self.by_category("ceramic")

    def add(self, material: Material) -> Material:
        if material.name in self._materials:
            raise KeyError(f"material {material.name!r} already exists")
        self._materials[material.name] = material
        self._alias_map[material.name.lower()] = material.name
        for alias in material.aliases:
            self._alias_map[alias.lower()] = material.name
        return material

    def to_dict(self) -> dict[str, Any]:
        return {name: m.to_dict() for name, m in self._materials.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterialDatabase:
        return cls({str(k): Material.from_dict(v) for k, v in data.items()})


__all__ = ["MATERIALS", "MATERIAL_CATEGORIES", "Material", "MaterialDatabase", "MaterialProperty"]
