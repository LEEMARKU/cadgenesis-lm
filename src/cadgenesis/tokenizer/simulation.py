"""
cadgenesis.tokenizer.simulation
================================
Simulation and physics token family.

Purpose
-------
Simulation tokens enable the model to reason about, specify, and interpret
engineering simulation results.  These cover:
- FEA (Finite Element Analysis) — structural, thermal, fatigue
- CFD (Computational Fluid Dynamics)
- Motion / kinematic simulation
- Electromagnetic simulation
- Simulation boundary conditions and loads
- Result interpretation tokens

Every simulation block is wrapped in <sim_start> ... <sim_end> special tokens
and uses SIM_ prefix.
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Simulation token lists
# ---------------------------------------------------------------------------

_FEA_STRUCTURAL: list[tuple[str, str]] = [
    ("SIM_FEA_STATIC", "Static structural FEA study"),
    ("SIM_FEA_MODAL", "Modal / natural frequency analysis"),
    ("SIM_FEA_BUCKLING", "Linear buckling analysis"),
    ("SIM_FEA_FATIGUE", "Fatigue life analysis"),
    ("SIM_FEA_NONLIN", "Nonlinear structural FEA (large deformation)"),
    ("SIM_FEA_CONTACT", "Contact mechanics FEA"),
    ("SIM_FEA_EXPLICIT", "Explicit dynamics FEA (impact / crash)"),
]

_FEA_THERMAL: list[tuple[str, str]] = [
    ("SIM_THERM_STEADY", "Steady-state heat transfer analysis"),
    ("SIM_THERM_TRANSIENT", "Transient heat transfer analysis"),
    ("SIM_THERM_COUPLED", "Thermo-structural coupled analysis"),
    ("SIM_THERM_RADIATION", "Radiation heat transfer analysis"),
]

_CFD: list[tuple[str, str]] = [
    ("SIM_CFD_STEADY", "Steady-state CFD (incompressible)"),
    ("SIM_CFD_TRANSIENT", "Transient CFD"),
    ("SIM_CFD_COMPRESS", "Compressible flow CFD"),
    ("SIM_CFD_MULTIPHASE", "Multiphase flow CFD"),
    ("SIM_CFD_THERMAL", "Conjugate heat transfer CFD"),
    ("SIM_CFD_TURBULENCE", "Turbulence model (RANS, LES, …)"),
]

_MOTION: list[tuple[str, str]] = [
    ("SIM_MBD_KINEMATIC", "Kinematic motion simulation (no dynamics)"),
    ("SIM_MBD_DYNAMIC", "Multi-body dynamics simulation"),
    ("SIM_MBD_RIGID", "Rigid body dynamics"),
    ("SIM_MBD_FLEXIBLE", "Flexible body dynamics"),
    ("SIM_MBD_CONTACT", "Contact / collision in motion simulation"),
]

_BOUNDARY_CONDITIONS: list[tuple[str, str]] = [
    # Structural BC
    ("SIM_BC_FIXED", "Fixed support (zero displacement at face/edge)"),
    ("SIM_BC_PINNED", "Pinned support (zero translation, free rotation)"),
    ("SIM_BC_ROLLER", "Roller support (free in one direction)"),
    ("SIM_BC_SYMMETRY", "Symmetry boundary condition"),
    ("SIM_BC_PERIODIC", "Periodic boundary condition"),
    # Loads
    ("SIM_LOAD_FORCE", "Applied force [→ NUM_xxx N]"),
    ("SIM_LOAD_PRESSURE", "Applied pressure [→ NUM_xxx MPa]"),
    ("SIM_LOAD_MOMENT", "Applied moment / torque [→ NUM_xxx N·m]"),
    ("SIM_LOAD_GRAVITY", "Gravitational body force"),
    ("SIM_LOAD_CENTRIFUG", "Centrifugal / rotational body force"),
    ("SIM_LOAD_THERMAL", "Thermal load (temperature field)"),
    ("SIM_LOAD_DISP", "Prescribed displacement [→ NUM_xxx mm]"),
    # Thermal BC
    ("SIM_THBC_TEMP", "Fixed temperature BC [→ NUM_xxx °C]"),
    ("SIM_THBC_FLUX", "Heat flux BC [→ NUM_xxx W/m²]"),
    ("SIM_THBC_CONVECT", "Convection BC (h coefficient [→ NUM_xxx])"),
    # CFD BC
    ("SIM_CFDBC_INLET", "CFD inlet boundary"),
    ("SIM_CFDBC_OUTLET", "CFD outlet boundary"),
    ("SIM_CFDBC_WALL", "CFD wall boundary (no-slip)"),
    ("SIM_CFDBC_SYMM", "CFD symmetry boundary"),
]

_RESULTS: list[tuple[str, str]] = [
    # Result type tokens (what was computed / reported)
    ("SIM_RES_STRESS", "Von Mises stress result"),
    ("SIM_RES_STRAIN", "Equivalent strain result"),
    ("SIM_RES_DISP", "Displacement magnitude result"),
    ("SIM_RES_SAFETY", "Factor of safety result"),
    ("SIM_RES_TEMP", "Temperature distribution result"),
    ("SIM_RES_HEAT_FLUX", "Heat flux distribution result"),
    ("SIM_RES_FREQ", "Natural frequency result [→ NUM_xxx Hz]"),
    ("SIM_RES_BUCKLE", "Buckling load factor result"),
    ("SIM_RES_FATIGUE_LIFE", "Fatigue life (cycles) result"),
    ("SIM_RES_VELOCITY", "Fluid velocity result [→ NUM_xxx m/s]"),
    ("SIM_RES_PRESSURE", "Fluid pressure result [→ NUM_xxx Pa]"),
    ("SIM_RES_DRAG", "Drag force result [→ NUM_xxx N]"),
    ("SIM_RES_MASS", "Part mass result [→ NUM_xxx kg]"),
    ("SIM_RES_COG", "Centre of gravity coordinates"),
    ("SIM_RES_INERTIA", "Moment of inertia result"),
    # Pass / fail
    ("SIM_RES_PASS", "Simulation result: design passes criteria"),
    ("SIM_RES_FAIL", "Simulation result: design fails criteria"),
    ("SIM_RES_MARGINAL", "Simulation result: design is marginal"),
    ("SIM_RES_CONVERGE", "Solver converged successfully"),
    ("SIM_RES_DIVERGE", "Solver did not converge"),
]

_SIM_UTILITY: list[tuple[str, str]] = [
    ("SIM_BEGIN", "Begin simulation specification block"),
    ("SIM_END", "End simulation specification block"),
    ("SIM_MESH_COARSE", "Coarse mesh quality"),
    ("SIM_MESH_MEDIUM", "Medium mesh quality"),
    ("SIM_MESH_FINE", "Fine mesh quality"),
    ("SIM_MESH_ADAPTIVE", "Adaptive mesh refinement"),
    ("SIM_SOLVER", "Solver specification follows"),
    ("SIM_UNITS_SI", "SI unit system"),
    ("SIM_UNITS_IMPERIAL", "Imperial unit system"),
]

_ALL_SIMULATION_TOKENS: list[tuple[str, str]] = (
    _FEA_STRUCTURAL + _FEA_THERMAL + _CFD + _MOTION + _BOUNDARY_CONDITIONS + _RESULTS + _SIM_UTILITY
)


# ---------------------------------------------------------------------------
# SimulationTokenizer
# ---------------------------------------------------------------------------


class SimulationTokenizer:
    """Registers all simulation tokens into a CADVocabulary."""

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        for token_str, desc in _ALL_SIMULATION_TOKENS:
            vocab.register(token_str, TokenFamily.SIMULATION, desc)

    @classmethod
    def all_token_strings(cls) -> list[str]:
        return [t for t, _ in _ALL_SIMULATION_TOKENS]

    @classmethod
    def fea_tokens(cls) -> list[str]:
        return [t for t, _ in (_FEA_STRUCTURAL + _FEA_THERMAL)]

    @classmethod
    def cfd_tokens(cls) -> list[str]:
        return [t for t, _ in _CFD]

    @classmethod
    def boundary_condition_tokens(cls) -> list[str]:
        return [t for t, _ in _BOUNDARY_CONDITIONS]

    @classmethod
    def result_tokens(cls) -> list[str]:
        return [t for t, _ in _RESULTS]
