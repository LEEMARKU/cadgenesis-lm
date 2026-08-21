"""
cadgenesis.tokenizer.material
==============================
Material and physical property token family.

Purpose
-------
Material tokens encode the material identity and physical properties of
CAD bodies.  They allow the model to understand the material context of
a design, which is critical for:
- Manufacturing process selection (e.g. aluminium → CNC milling vs.
  ABS → injection moulding)
- FEA simulation (density, Young's modulus, thermal conductivity)
- Cost estimation (material price per kg)

Token categories:
1. Metal alloys
2. Polymers (thermoplastics, thermosets, elastomers)
3. Composites and advanced materials
4. Ceramics and glasses
5. Physical property tokens (followed by NUM_xxx values)
6. Surface finish / treatment tokens
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Material token lists
# ---------------------------------------------------------------------------

_METALS: list[tuple[str, str]] = [
    # Steels
    ("MAT_STEEL_MILD", "Mild / low-carbon steel (e.g. AISI 1018)"),
    ("MAT_STEEL_MEDIUM", "Medium-carbon steel (e.g. AISI 1045)"),
    ("MAT_STEEL_HIGH", "High-carbon / tool steel (e.g. D2, H13)"),
    ("MAT_STEEL_SS_304", "Stainless steel 304 (austenitic)"),
    ("MAT_STEEL_SS_316", "Stainless steel 316 (marine grade)"),
    ("MAT_STEEL_SS_17_4", "Stainless steel 17-4 PH (precipitation hardened)"),
    ("MAT_STEEL_ALLOY_4140", "Alloy steel 4140 (Cr-Mo)"),
    ("MAT_STEEL_SPRING", "Spring steel"),
    ("MAT_STEEL_BEARING", "Bearing steel (e.g. 52100)"),
    # Aluminium
    ("MAT_AL_6061", "Aluminium alloy 6061-T6 (general purpose)"),
    ("MAT_AL_7075", "Aluminium alloy 7075-T6 (aerospace)"),
    ("MAT_AL_2024", "Aluminium alloy 2024 (fatigue resistant)"),
    ("MAT_AL_5052", "Aluminium alloy 5052 (marine, sheet metal)"),
    ("MAT_AL_CAST", "Cast aluminium alloy"),
    # Titanium
    ("MAT_TI_6AL4V", "Titanium Ti-6Al-4V Grade 5 (aerospace)"),
    ("MAT_TI_PURE", "Pure titanium Grade 2 (corrosion resistant)"),
    # Copper and alloys
    ("MAT_COPPER", "Pure copper (electrical)"),
    ("MAT_BRASS", "Brass (Cu-Zn alloy)"),
    ("MAT_BRONZE", "Bronze (Cu-Sn alloy)"),
    ("MAT_BERYLLIUM_CU", "Beryllium copper (springs, contacts)"),
    # Other metals
    ("MAT_MAGNESIUM", "Magnesium alloy (lightweight)"),
    ("MAT_ZINC", "Zinc die-cast alloy"),
    ("MAT_NICKEL", "Nickel superalloy (e.g. Inconel 718)"),
    ("MAT_CAST_IRON", "Cast iron (grey, ductile)"),
]

_POLYMERS: list[tuple[str, str]] = [
    # Thermoplastics
    ("MAT_ABS", "ABS (Acrylonitrile Butadiene Styrene)"),
    ("MAT_PLA", "PLA (Polylactic Acid, biodegradable)"),
    ("MAT_PETG", "PETG (impact-resistant, food-safe)"),
    ("MAT_NYLON_PA6", "Nylon PA6 (general purpose)"),
    ("MAT_NYLON_PA66", "Nylon PA66 (higher temperature)"),
    ("MAT_POM", "POM / Delrin (acetal, low friction)"),
    ("MAT_PEEK", "PEEK (high-temperature, chemical resistant)"),
    ("MAT_POLYCARBONATE", "Polycarbonate (impact resistant, transparent)"),
    ("MAT_POLYPROPYLENE", "Polypropylene PP (chemical resistant)"),
    ("MAT_POLYETHYLENE_HD", "HDPE (high-density polyethylene)"),
    ("MAT_POLYETHYLENE_LD", "LDPE (flexible packaging)"),
    ("MAT_PVC", "PVC (rigid / flexible)"),
    ("MAT_PTFE", "PTFE / Teflon (non-stick, low friction)"),
    ("MAT_ULTEM", "Ultem / PEI (high-temp engineering polymer)"),
    # Thermosets
    ("MAT_EPOXY", "Epoxy resin (adhesive, laminating)"),
    ("MAT_POLYURETHANE", "Polyurethane (foam, coating, elastomer)"),
    ("MAT_PHENOLIC", "Phenolic resin (Bakelite, electrical)"),
    # Elastomers
    ("MAT_SILICONE", "Silicone rubber (high-temp, biocompatible)"),
    ("MAT_NITRILE", "Nitrile rubber (oil resistant seals)"),
    ("MAT_EPDM", "EPDM rubber (weathering, outdoor)"),
]

_COMPOSITES: list[tuple[str, str]] = [
    ("MAT_CFRP", "Carbon-fibre reinforced polymer"),
    ("MAT_GFRP", "Glass-fibre reinforced polymer"),
    ("MAT_ARAMID", "Aramid / Kevlar fibre composite"),
    ("MAT_BASALT", "Basalt fibre composite"),
    ("MAT_MMC_AL_SIC", "Metal-matrix composite: Al-SiC"),
    ("MAT_WOOD", "Engineering wood / timber"),
    ("MAT_PLYWOOD", "Plywood"),
    ("MAT_CONCRETE", "Concrete (structural)"),
    ("MAT_REINFORCED_CONC", "Reinforced concrete (rebar)"),
]

_CERAMICS: list[tuple[str, str]] = [
    ("MAT_ALUMINA", "Aluminium oxide (Al₂O₃) ceramic"),
    ("MAT_ZIRCONIA", "Zirconia (ZrO₂, wear resistant)"),
    ("MAT_SIC_CERAMIC", "Silicon carbide ceramic"),
    ("MAT_SI3N4", "Silicon nitride ceramic"),
    ("MAT_GLASS", "Borosilicate glass"),
    ("MAT_SODA_GLASS", "Soda-lime glass"),
]

_PROPERTY_TOKENS: list[tuple[str, str]] = [
    # Physical property declarations (each followed by a NUM_xxx token)
    ("MAT_PROP_DENSITY", "Material density (kg/m³) [→ NUM_xxx]"),
    ("MAT_PROP_YOUNGS", "Young's modulus (GPa) [→ NUM_xxx]"),
    ("MAT_PROP_POISSON", "Poisson's ratio (dimensionless) [→ NUM_xxx]"),
    ("MAT_PROP_YIELD", "Yield strength (MPa) [→ NUM_xxx]"),
    ("MAT_PROP_UTS", "Ultimate tensile strength (MPa) [→ NUM_xxx]"),
    ("MAT_PROP_HARDNESS", "Hardness (HRC / HRB / HV) [→ NUM_xxx]"),
    ("MAT_PROP_THERMAL_K", "Thermal conductivity (W/m·K) [→ NUM_xxx]"),
    ("MAT_PROP_CTE", "Coefficient of thermal expansion (µm/m·K)"),
    ("MAT_PROP_MELTING", "Melting / glass-transition temperature (°C)"),
    ("MAT_PROP_COST", "Material cost (USD/kg) [→ NUM_xxx]"),
]

_SURFACE_FINISH: list[tuple[str, str]] = [
    ("MAT_FINISH_RAW", "As-machined / raw surface"),
    ("MAT_FINISH_POLISH", "Polished surface"),
    ("MAT_FINISH_ANODIZE", "Anodized (aluminium)"),
    ("MAT_FINISH_POWDER", "Powder coated"),
    ("MAT_FINISH_PAINT", "Painted / wet-coated"),
    ("MAT_FINISH_PLATED_ZN", "Zinc plated (corrosion protection)"),
    ("MAT_FINISH_PLATED_NI", "Nickel plated"),
    ("MAT_FINISH_PLATED_CR", "Hard chrome plated"),
    ("MAT_FINISH_CARBURIZE", "Carburized / case-hardened"),
    ("MAT_FINISH_NITRIDE", "Nitrided"),
    ("MAT_FINISH_BEAD_BLAST", "Bead blasted"),
    ("MAT_FINISH_VIBRATORY", "Vibratory tumbled"),
    ("MAT_FINISH_PASSIVATE", "Passivated stainless steel"),
]

_MATERIAL_UTILITY: list[tuple[str, str]] = [
    ("MAT_BEGIN", "Begin material specification block"),
    ("MAT_END", "End material specification block"),
    ("MAT_CUSTOM", "Custom / user-defined material"),
    ("MAT_UNKNOWN", "Material unspecified"),
]

_ALL_MATERIAL_TOKENS: list[tuple[str, str]] = (
    _METALS
    + _POLYMERS
    + _COMPOSITES
    + _CERAMICS
    + _PROPERTY_TOKENS
    + _SURFACE_FINISH
    + _MATERIAL_UTILITY
)


# ---------------------------------------------------------------------------
# MaterialTokenizer
# ---------------------------------------------------------------------------


class MaterialTokenizer:
    """Registers all material tokens into a CADVocabulary."""

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        for token_str, desc in _ALL_MATERIAL_TOKENS:
            vocab.register(token_str, TokenFamily.MATERIAL, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_MATERIAL_TOKENS]

    @classmethod
    def metal_tokens(cls) -> list[str]:
        return [t for t, _ in _METALS]

    @classmethod
    def polymer_tokens(cls) -> list[str]:
        return [t for t, _ in _POLYMERS]

    @classmethod
    def property_tokens(cls) -> list[str]:
        return [t for t, _ in _PROPERTY_TOKENS]

    @classmethod
    def finish_tokens(cls) -> list[str]:
        return [t for t, _ in _SURFACE_FINISH]
