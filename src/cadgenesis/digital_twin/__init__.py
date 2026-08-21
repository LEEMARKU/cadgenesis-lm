"""cadgenesis.digital_twin
=========================
Minimal industrial digital-twin package for CADGenesis-LM v6.0.

Materializes design parts into a twin record (mesh + B-Rep + analytic
properties), keeps them in sync with execution results, runs simulations
against the twin, persists snapshots to memory, and reports drift/status.
"""

from cadgenesis.digital_twin.twin import DigitalTwinSystem, TwinRecord

__all__ = ["DigitalTwinSystem", "TwinRecord"]
