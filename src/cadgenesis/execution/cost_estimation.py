"""cadgenesis.execution.cost_estimation
=====================================
Cost estimation for the CAD execution pipeline.

Material, machining, additive-print, assembly and amortized tooling cost
breakdown from part descriptors, plus the legacy token-prefix cost mapping
(BOX → $25, CYLINDER → $35, else $50) preserved for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class CostBreakdown:
    """Cost breakdown with a total."""

    material_usd: float = 0.0
    machining_usd: float = 0.0
    printing_usd: float = 0.0
    assembly_usd: float = 0.0
    tooling_usd: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.material_usd
            + self.machining_usd
            + self.printing_usd
            + self.assembly_usd
            + self.tooling_usd
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "material_usd": round(self.material_usd, 4),
            "machining_usd": round(self.machining_usd, 4),
            "printing_usd": round(self.printing_usd, 4),
            "assembly_usd": round(self.assembly_usd, 4),
            "tooling_usd": round(self.tooling_usd, 4),
            "total": round(self.total, 4),
        }


class CostEstimator:
    """Cost estimator for part descriptors.

    ``part`` keys: ``volume_m3`` (or ``volume_mm3``), ``material`` (dict with
    ``name`` and/or ``density_kg_m3``), ``processes`` (list), ``batch_size``,
    ``feature_count``, ``part_count``.  Material price tables are configurable
    per material name.
    """

    DEFAULT_PRICES_USD_PER_KG = {
        "steel": 2.5,
        "stainless": 6.0,
        "aluminum": 4.0,
        "titanium": 60.0,
        "brass": 8.0,
        "copper": 9.0,
        "plastic": 3.0,
        "abs": 3.0,
        "pla": 2.0,
        "nylon": 5.0,
        "carbon": 25.0,
    }

    def __init__(self, prices_usd_per_kg: dict[str, float] | None = None) -> None:
        merged = dict(self.DEFAULT_PRICES_USD_PER_KG)
        if prices_usd_per_kg:
            merged.update(prices_usd_per_kg)
        self.prices_usd_per_kg = merged

    def estimate(self, part: dict[str, Any]) -> CostBreakdown:
        """Breakdown estimate from a part descriptor (never raises)."""
        volume_m3 = part.get("volume_m3")
        if volume_m3 is None and "volume_mm3" in part:
            volume_m3 = float(part["volume_mm3"]) * 1e-9
        material = part.get("material") or {}
        name = str(material.get("name") or "").lower()
        price = self.prices_usd_per_kg.get(name, 10.0)
        density = float(material.get("density_kg_m3") or 0.0)
        processes = [str(p).lower() for p in (part.get("processes") or [])]
        batch = max(1, int(part.get("batch_size") or 1))
        feature_count = int(part.get("feature_count") or 1)
        part_count = int(part.get("part_count") or 1)

        material_usd = 0.0
        if volume_m3 is not None and volume_m3 > 0.0:
            mass = volume_m3 * (density if density > 0.0 else 7800.0)
            material_usd = mass * price

        machining_usd = 0.0
        if any(p in ("machining", "cnc", "cnc_milling", "cnc_turning") for p in processes):
            machining_usd = 3.0 + 2.0 * feature_count
            if volume_m3 is not None:
                machining_usd += max(0.0, (volume_m3 * 1e6) / 20000.0) * 1.5

        printing_usd = 0.0
        if any("print" in p for p in processes):
            print_hours = volume_m3 * 1000000.0 / 12000.0 if volume_m3 is not None else 0.5
            printing_usd = print_hours * 8.0

        assembly_usd = 0.0
        if any(p in ("assembly", "welding") for p in processes):
            assembly_usd = 2.5 * part_count

        tooling_usd = 0.0
        if any(p in ("casting", "injection_molding", "die") for p in processes):
            tooling_usd = 5000.0 / batch

        return CostBreakdown(
            material_usd=material_usd,
            machining_usd=machining_usd,
            printing_usd=printing_usd,
            assembly_usd=assembly_usd,
            tooling_usd=tooling_usd,
        )

    def token_cost(self, cad_tokens: Sequence[str]) -> float:
        """Legacy token-prefix cost mapping (BOX $25 / CYLINDER $35 / else $50)."""
        head = cad_tokens[0].upper() if cad_tokens else "UNKNOWN"
        if "BOX" in head:
            return 25.0
        if "CYLINDER" in head:
            return 35.0
        return 50.0

    def summary(self) -> dict[str, Any]:
        return {"materials": sorted(self.prices_usd_per_kg)}


__all__ = ["CostBreakdown", "CostEstimator"]
