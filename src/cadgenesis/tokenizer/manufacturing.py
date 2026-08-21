"""
cadgenesis.tokenizer.manufacturing
====================================
Manufacturing process token family.

Purpose
-------
Manufacturing tokens encode how a part will be physically produced.  This
enables the model to reason about design-for-manufacturing (DFM) — generating
CAD that is not just geometrically valid but producible by a specific process.

Token categories:
1. Primary manufacturing processes (machining, forming, additive, casting)
2. Secondary / finishing processes
3. Process parameters (feeds, speeds, tool types)
4. Quality / inspection tokens
5. Cost / scheduling tokens
6. Utility tokens
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Manufacturing token lists
# ---------------------------------------------------------------------------

_MACHINING: list[tuple[str, str]] = [
    ("MFG_CNC_MILL", "CNC milling (3-axis)"),
    ("MFG_CNC_MILL_5X", "5-axis CNC milling"),
    ("MFG_CNC_TURN", "CNC turning (lathe)"),
    ("MFG_CNC_TURN_MILL", "Turn-mill (mill-turn machining centre)"),
    ("MFG_DRILL", "Drilling operation"),
    ("MFG_BORE", "Boring (precision hole enlarging)"),
    ("MFG_REAM", "Reaming (finishing a drilled hole)"),
    ("MFG_TAP", "Tapping (threading a hole)"),
    ("MFG_GRIND", "Grinding (surface / cylindrical)"),
    ("MFG_EDM_WIRE", "Wire EDM (electrical discharge machining)"),
    ("MFG_EDM_SINK", "Sinker / die-sinking EDM"),
    ("MFG_LASER_CUT", "Laser cutting (sheet metal / flat stock)"),
    ("MFG_WATERJET", "Waterjet cutting"),
    ("MFG_PLASMA", "Plasma cutting"),
    ("MFG_BROACH", "Broaching"),
    ("MFG_HOB", "Gear hobbing"),
]

_FORMING: list[tuple[str, str]] = [
    ("MFG_SHEET_BEND", "Sheet metal bending (brake press)"),
    ("MFG_SHEET_STAMP", "Sheet metal stamping / blanking"),
    ("MFG_DEEP_DRAW", "Deep drawing"),
    ("MFG_ROLL_FORM", "Roll forming"),
    ("MFG_HYDROFORM", "Hydroforming"),
    ("MFG_FORGE_HOT", "Hot forging"),
    ("MFG_FORGE_COLD", "Cold forging"),
    ("MFG_EXTRUSION", "Extrusion (metal / plastic profile)"),
    ("MFG_DRAW_WIRE", "Wire / tube drawing"),
    ("MFG_SPIN", "Metal spinning / flow forming"),
]

_CASTING: list[tuple[str, str]] = [
    ("MFG_SAND_CAST", "Sand casting"),
    ("MFG_DIE_CAST", "Die casting"),
    ("MFG_INVEST_CAST", "Investment / lost-wax casting"),
    ("MFG_PERM_MOLD", "Permanent mould casting"),
    ("MFG_CENTRIFUGAL", "Centrifugal casting"),
    ("MFG_INJECTION_MOLD", "Injection moulding (plastic)"),
    ("MFG_BLOW_MOLD", "Blow moulding"),
    ("MFG_ROTO_MOLD", "Rotational moulding"),
    ("MFG_COMPRESSION_MOL", "Compression moulding (thermosets)"),
    ("MFG_RESIN_TRANSFER", "Resin transfer moulding (RTM)"),
]

_ADDITIVE: list[tuple[str, str]] = [
    ("MFG_AM_FDM", "Fused deposition modelling (FDM/FFF)"),
    ("MFG_AM_SLA", "Stereolithography (SLA)"),
    ("MFG_AM_SLS", "Selective laser sintering (SLS)"),
    ("MFG_AM_DMLS", "Direct metal laser sintering (DMLS/SLM)"),
    ("MFG_AM_BINDER", "Binder jetting"),
    ("MFG_AM_MJF", "Multi-jet fusion (MJF)"),
    ("MFG_AM_DED", "Directed energy deposition (DED)"),
    ("MFG_AM_EBM", "Electron beam melting (EBM)"),
    ("MFG_AM_WIRE_ARC", "Wire arc additive manufacturing (WAAM)"),
    ("MFG_AM_BUILD_DIR", "Additive manufacturing build direction"),
    ("MFG_AM_SUPPORT", "Support structure required"),
    ("MFG_AM_NO_SUPPORT", "No support structure required"),
]

_JOINING: list[tuple[str, str]] = [
    ("MFG_WELD_MIG", "MIG / GMAW welding"),
    ("MFG_WELD_TIG", "TIG / GTAW welding"),
    ("MFG_WELD_SPOT", "Spot welding (resistance)"),
    ("MFG_WELD_LASER", "Laser welding"),
    ("MFG_WELD_BRAZE", "Brazing"),
    ("MFG_WELD_SOLDER", "Soldering"),
    ("MFG_BOND_ADHESIVE", "Adhesive bonding"),
    ("MFG_FASTEN_BOLT", "Bolted fastener joint"),
    ("MFG_FASTEN_RIVET", "Riveted joint"),
    ("MFG_FASTEN_PRESS", "Press fit (interference fit)"),
    ("MFG_FASTEN_SNAP", "Snap fit (integral plastic clip)"),
]

_PROCESS_PARAMS: list[tuple[str, str]] = [
    # Machining parameters (each followed by NUM_xxx)
    ("MFG_PARAM_FEED", "Feed rate [→ NUM_xxx mm/rev or mm/min]"),
    ("MFG_PARAM_SPEED", "Cutting speed / RPM [→ NUM_xxx]"),
    ("MFG_PARAM_DOC", "Depth of cut [→ NUM_xxx mm]"),
    ("MFG_PARAM_TOOL_DIA", "Tool diameter [→ NUM_xxx mm]"),
    ("MFG_PARAM_TOL", "Manufacturing tolerance [→ NUM_xxx mm]"),
    ("MFG_PARAM_RA", "Surface roughness Ra [→ NUM_xxx µm]"),
    ("MFG_PARAM_BATCH", "Batch / production quantity [→ NUM_xxx]"),
    ("MFG_PARAM_LEADTIME", "Lead time [→ NUM_xxx days]"),
    ("MFG_PARAM_COST", "Estimated unit cost [→ NUM_xxx USD]"),
]

_QUALITY: list[tuple[str, str]] = [
    ("MFG_QC_CMM", "CMM (coordinate measuring machine) inspection"),
    ("MFG_QC_VISUAL", "Visual inspection"),
    ("MFG_QC_NDT_UT", "Ultrasonic NDT inspection"),
    ("MFG_QC_NDT_RT", "Radiographic NDT inspection"),
    ("MFG_QC_NDT_MT", "Magnetic particle NDT"),
    ("MFG_QC_NDT_PT", "Dye penetrant NDT"),
    ("MFG_QC_PASS", "Quality check: pass"),
    ("MFG_QC_FAIL", "Quality check: fail / reject"),
    ("MFG_QC_REWORK", "Rework required"),
    ("MFG_QC_CERT", "Material certification required"),
]

_MFG_UTILITY: list[tuple[str, str]] = [
    ("MFG_BEGIN", "Begin manufacturing specification block"),
    ("MFG_END", "End manufacturing specification block"),
    ("MFG_UNKNOWN", "Manufacturing process not specified"),
    ("MFG_MULTI", "Multi-process part (machining + additive, etc.)"),
]

_ALL_MANUFACTURING_TOKENS: list[tuple[str, str]] = (
    _MACHINING
    + _FORMING
    + _CASTING
    + _ADDITIVE
    + _JOINING
    + _PROCESS_PARAMS
    + _QUALITY
    + _MFG_UTILITY
)


# ---------------------------------------------------------------------------
# ManufacturingTokenizer
# ---------------------------------------------------------------------------


class ManufacturingTokenizer:
    """Registers all manufacturing tokens into a CADVocabulary."""

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        for token_str, desc in _ALL_MANUFACTURING_TOKENS:
            vocab.register(token_str, TokenFamily.MANUFACTURING, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_MANUFACTURING_TOKENS]

    @classmethod
    def additive_tokens(cls) -> list[str]:
        return [t for t, _ in _ADDITIVE]

    @classmethod
    def machining_tokens(cls) -> list[str]:
        return [t for t, _ in _MACHINING]

    @classmethod
    def casting_tokens(cls) -> list[str]:
        return [t for t, _ in _CASTING]

    @classmethod
    def joining_tokens(cls) -> list[str]:
        return [t for t, _ in _JOINING]
