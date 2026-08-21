"""cadgenesis.cad.manufacturing
=============================
Manufacturing features (CNC, 3D printing, casting, injection moulding, sheet
metal, welding) and a process-selection engine.
"""

from cadgenesis.cad.manufacturing.features import (
    PROCESS_GROUPS,
    ManufacturingFeature,
    casting_feature,
    cnc_feature,
    injection_feature,
    make_feature,
    print_feature,
    sheet_metal_feature,
    welding_feature,
)
from cadgenesis.cad.manufacturing.process import (
    ProcessSelection,
    ProcessSelector,
    ProcessSuggestion,
)

__all__ = [
    "PROCESS_GROUPS",
    "ManufacturingFeature",
    "ProcessSelection",
    "ProcessSelector",
    "ProcessSuggestion",
    "casting_feature",
    "cnc_feature",
    "injection_feature",
    "make_feature",
    "print_feature",
    "sheet_metal_feature",
    "welding_feature",
]
