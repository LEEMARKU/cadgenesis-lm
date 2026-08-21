"""cadgenesis.cad.integration
============================
Integration layer linking the new ``cad`` package with the existing
tokenizer, transformer, memory, reasoning, execution and simulation subsystems.

Modules:
- :mod:`tokenizer_bridge`    — CAD objects <-> CAD token strings / sequences
- :mod:`transformer_bridge`  — CAD designs <-> transformer batches/tensors
- :mod:`memory_bridge`       — store / retrieve designs in CADMemory
- :mod:`reasoning_bridge`    — CAD objects <-> reasoning toolkit inputs
- :mod:`execution_bridge`    — CAD objects -> execution engine -> feedback
- :mod:`simulation_bridge`   — CAD meshes/designs <-> simulation memory
- :mod:`pipeline`            — end-to-end validate -> reason -> tokenize -> store
"""

from cadgenesis.cad.integration.execution_bridge import ExecutionBridge
from cadgenesis.cad.integration.memory_bridge import CADMemoryBridge
from cadgenesis.cad.integration.pipeline import CADIntelligencePipeline, PipelineResult
from cadgenesis.cad.integration.reasoning_bridge import ReasoningBridge, bridge
from cadgenesis.cad.integration.simulation_bridge import SimulationBridge, SimulationSetup
from cadgenesis.cad.integration.tokenizer_bridge import (
    FEATURE_TOKEN_MAP,
    MATERIAL_TOKEN_MAP,
    TokenizerBridge,
)
from cadgenesis.cad.integration.transformer_bridge import TransformerBridge

__all__ = [
    "FEATURE_TOKEN_MAP",
    "MATERIAL_TOKEN_MAP",
    "CADIntelligencePipeline",
    "CADMemoryBridge",
    "ExecutionBridge",
    "PipelineResult",
    "ReasoningBridge",
    "SimulationBridge",
    "SimulationSetup",
    "TokenizerBridge",
    "TransformerBridge",
    "bridge",
]
