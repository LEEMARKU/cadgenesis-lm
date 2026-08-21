"""cadgenesis.tokenizer.simulation_tokens
=======================================
Simulation and physics token definitions.

Canonical public surface over :mod:`cadgenesis.tokenizer.simulation`
exposing the complete simulation token table used by the CAD vocabulary.
"""

from cadgenesis.tokenizer.simulation import (
    _ALL_SIMULATION_TOKENS,
    SimulationTokenizer,
)

ALL_SIMULATION_TOKENS = _ALL_SIMULATION_TOKENS

__all__ = ["ALL_SIMULATION_TOKENS", "SimulationTokenizer"]
