"""
cadgenesis.agents.multi_agent_system
====================================
Internal Multi-Agent Transformer System (Embedded in Model) for CADGenesis-LM v2.0:
1. Planner Agent — overall CAD sequence planning & intent understanding
2. Geometry Agent — B-Rep solid & sketch primitive construction
3. Constraint Agent — parametric & geometric constraint reasoning
4. Manufacturing Agent — Design for Manufacturing (DFM) verification
5. Validation Agent — topological & geometric validation
6. Optimization Agent — mass, stress, & parameter optimization
7. Assembly Agent — multi-component mating & joint kinematics
8. Simulation Agent — FEA/CFD load case & boundary condition reasoning

Shared-Memory Integration
-------------------------
Every agent conditions on the Layer-Integrated Memory Pools: when a combined
memory bank ``(B, total_slots, C)`` is passed to :meth:`MultiAgentSystem.forward`,
it is mean-pooled, projected, and gated into each agent's input, so each role
reads shared memory before emitting its view of the hidden states.  The
transformer's decoder passes the evolving memory bank on every block.

Complexity
----------
    forward (no memory): O(B · T · C · d)   (8 agents + bus projection)
    forward (+memory):   + O(B · C · d)     (pool + project + gate)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InternalAgentRole(nn.Module):
    """Specific agent role head within the multi-agent transformer system."""

    def __init__(self, role_name: str, d_model: int):
        super().__init__()
        self.role_name = role_name
        self.d_model = d_model
        self.role_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.agent_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, T, C)
        Returns agent-processed hidden states: (B, T, C)
        """
        prompted = hidden_states + self.role_embedding
        return self.agent_transform(prompted)


class MultiAgentSystem(nn.Module):
    """
    Internal 8-Agent System embedded directly within the Transformer backbone.
    Communicates via a shared Agent Communication Bus and conditions on the
    Layer-Integrated Memory Pools when a memory bank is provided.
    """

    ROLES: list[str] = [
        "planner",
        "geometry",
        "constraint",
        "manufacturing",
        "validation",
        "optimization",
        "assembly",
        "simulation",
    ]

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model

        self.planner_agent = InternalAgentRole("planner", d_model)
        self.geometry_agent = InternalAgentRole("geometry", d_model)
        self.constraint_agent = InternalAgentRole("constraint", d_model)
        self.mfg_agent = InternalAgentRole("manufacturing", d_model)
        self.validation_agent = InternalAgentRole("validation", d_model)
        self.optimization_agent = InternalAgentRole("optimization", d_model)
        self.assembly_agent = InternalAgentRole("assembly", d_model)
        self.simulation_agent = InternalAgentRole("simulation", d_model)

        self.agent_bus_proj = nn.Linear(d_model * 8, d_model)

        # Shared-memory conditioning: pool → project → gate into agent inputs.
        self.memory_context_proj = nn.Linear(d_model, d_model)
        self.memory_gate = nn.Linear(d_model, 1)

    @property
    def agent_names(self) -> list[str]:
        """The 8 role names in bus order (diagnostics / reporting)."""
        return list(self.ROLES)

    def _memory_context(self, memory_bank: torch.Tensor) -> torch.Tensor:
        """
        Reduce the shared memory bank to a gated context vector (B, 1, C).
        """
        pooled = memory_bank.mean(dim=1)  # (B, C)
        proj = self.memory_context_proj(pooled).unsqueeze(1)  # (B, 1, C)
        gate = torch.sigmoid(self.memory_gate(pooled)).unsqueeze(-1)  # (B, 1, 1)
        return gate * proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        hidden_states: (B, T, C)
        memory_bank: (B, total_slots, C), optional — shared memory pools.

        Returns: aggregated agent communication bus tensor (B, T, C)
        """
        agent_input = hidden_states
        if memory_bank is not None:
            agent_input = hidden_states + self._memory_context(memory_bank)

        out_p = self.planner_agent(agent_input)
        out_g = self.geometry_agent(agent_input)
        out_c = self.constraint_agent(agent_input)
        out_m = self.mfg_agent(agent_input)
        out_v = self.validation_agent(agent_input)
        out_o = self.optimization_agent(agent_input)
        out_a = self.assembly_agent(agent_input)
        out_s = self.simulation_agent(agent_input)

        concat_states = torch.cat([out_p, out_g, out_c, out_m, out_v, out_o, out_a, out_s], dim=-1)
        return self.agent_bus_proj(concat_states)
