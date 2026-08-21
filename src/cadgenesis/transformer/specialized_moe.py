"""
cadgenesis.transformer.specialized_moe
======================================
Specialized Mixture-of-Experts for CADGenesis-LM v6.0 (Pillar 1).

The generic :class:`cadgenesis.transformer.moe.SparseMoEFFN` routes tokens
across interchangeable experts.  :class:`SpecializedMoEFFN` additionally
*labels* every expert with an engineering domain so the routing signal becomes
semantically meaningful and load statistics can be reported per discipline:

* ``geometry``       — primitive, B-Rep and sketch reasoning.
* ``manufacturing``  — machining/process planning knowledge.
* ``reasoning``      — symbolic / neuro-symbolic deduction.
* ``simulation``     — FEA / physics-conditioned transforms.
* ``optimization``   — topology/parametric optimisation.

Each domain owns ``experts_per_domain`` expert instances.  The router assigns
every token to its ``top_k`` favourite experts *across all domains* (sparse
activation), and a Shazeer-style auxiliary load-balancing loss keeps the
routing uniform.  Domains can be added at runtime via
:func:`register_expert_type` without touching the core — the plugin hook used by
the Configurable Transformer Evolution framework.

Complexity
----------
    Time:   O(M · top_k · E · d²)   (E = experts per domain)
    Space:  O(M · top_k)
    Growth: O(experts_per_domain · d) per new domain (amortised).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DOMAIN_EXPERTS = (
    "geometry",
    "manufacturing",
    "reasoning",
    "simulation",
    "optimization",
)

# Mutable registry so researchers can extend the expert taxonomy at runtime.
_registered_domains: list[str] = list(DEFAULT_DOMAIN_EXPERTS)


def registered_expert_types() -> list[str]:
    """Snapshot of the currently registered expert domain names."""
    return list(_registered_domains)


def register_expert_type(name: str) -> None:
    """
    Register a new expert domain (plugin hook for the evolution framework).

    Existing :class:`SpecializedMoEFFN` modules are unaffected until they grow
    (their router is expanded lazily by :meth:`SpecializedMoEFFN.add_domain`).
    """
    name = name.strip().lower()
    if not name:
        raise ValueError("expert type name must be non-empty.")
    if name not in _registered_domains:
        _registered_domains.append(name)


class DomainExpert(nn.Module):
    """A single two-layer GELU expert belonging to a named engineering domain."""

    def __init__(self, domain: str, d_model: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.domain = domain
        self.w1 = nn.Linear(d_model, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M, d_model) → (M, d_model)"""
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class SpecializedMoEFFN(nn.Module):
    """
    Top-k routed MoE over *domain-labelled* experts with load-balancing.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    expert_types : tuple[str, ...] | list[str]
        Ordered domain names; each contributes ``experts_per_domain`` experts.
    experts_per_domain : int
        Number of expert instances per domain.
    top_k : int
        Number of experts activated per token across the whole pool.
    expert_dim : int | None
        Hidden width per expert (defaults to ``2 * d_model``).
    dropout : float
        Dropout inside each expert.
    router_jitter : float
        Uniform noise magnitude on router logits during training.
    """

    def __init__(
        self,
        d_model: int,
        expert_types: tuple[str, ...] | list[str] = DEFAULT_DOMAIN_EXPERTS,
        experts_per_domain: int = 2,
        top_k: int = 2,
        expert_dim: int | None = None,
        dropout: float = 0.1,
        router_jitter: float = 0.02,
    ):
        super().__init__()
        expert_types = tuple(expert_types)
        if not expert_types:
            raise ValueError("expert_types must not be empty.")
        if experts_per_domain < 1:
            raise ValueError("experts_per_domain must be >= 1.")
        total_experts = len(expert_types) * experts_per_domain
        if top_k < 1 or top_k > total_experts:
            raise ValueError(f"top_k must satisfy 1 <= top_k <= {total_experts} (total experts).")

        self.d_model = d_model
        self.expert_types = list(expert_types)
        self.experts_per_domain = experts_per_domain
        self.num_experts = total_experts
        self.top_k = top_k
        self.expert_dim = expert_dim or (2 * d_model)
        self.router_jitter = router_jitter

        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                DomainExpert(domain, d_model, self.expert_dim, dropout)
                for domain in self.expert_types
                for _ in range(experts_per_domain)
            ]
        )

        self._aux_loss: torch.Tensor
        self.register_buffer("_aux_loss", torch.zeros(()), persistent=False)
        self._aux_loss.zero_()

    # -------------------------------------------------------------- metadata

    def domain_of(self, expert_idx: int) -> str:
        """Domain label of a flat expert index."""
        if not (0 <= expert_idx < self.num_experts):
            raise IndexError(f"expert index {expert_idx} out of range.")
        return self.expert_types[expert_idx // self.experts_per_domain]

    def expert_domains(self) -> list[str]:
        """Domain label of every flat expert index."""
        return [self.domain_of(i) for i in range(self.num_experts)]

    # ---------------------------------------------------------------- growth

    def add_domain(self, domain: str) -> int:
        """
        Append ``experts_per_domain`` new experts for a (possibly novel) domain
        and expand the router, preserving existing routing behaviour.
        """
        if not domain:
            raise ValueError("domain must be non-empty.")
        added = self.experts_per_domain
        for _ in range(added):
            self.experts.append(DomainExpert(domain, self.d_model, self.expert_dim))
        self.expert_types.append(domain)
        self.num_experts += added

        new_router = nn.Linear(self.d_model, self.num_experts, bias=False)
        with torch.no_grad():
            new_router.weight[: self.num_experts - added].copy_(self.router.weight)
            new_router.weight[self.num_experts - added :].normal_(0.0, 0.02)
        self.router = new_router
        return self.num_experts

    # ------------------------------------------------------------- routing

    def _router_probs(self, flat: torch.Tensor) -> torch.Tensor:
        logits = self.router(flat)
        if self.training and self.router_jitter > 0.0:
            logits = logits + torch.empty_like(logits).uniform_(
                -self.router_jitter, self.router_jitter
            )
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) → (B, T, d_model).

        Only ``top_k`` experts are executed per token (sparse activation).
        Stores ``routing_decisions``, ``aux_loss_value`` and per-domain loads.
        """
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        M = flat.shape[0]

        probs = self._router_probs(flat)  # (M, E)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (M, k)
        norm = top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = top_probs / norm

        out = torch.zeros_like(flat)
        domain_counts = {d: 0 for d in self.expert_types}
        for k in range(self.top_k):
            selected = top_idx[:, k]
            w_k = weights[:, k]
            for e in range(self.num_experts):
                mask = selected == e
                if mask.any():
                    domain_counts[self.domain_of(e)] += int(mask.sum().item())
                    out[mask] = out[mask] + w_k[mask].unsqueeze(-1) * self.experts[e](flat[mask])

        token_fraction = torch.zeros(self.num_experts, device=x.device)
        token_fraction.index_add_(
            0, top_idx.reshape(-1), torch.ones_like(top_idx.reshape(-1), dtype=probs.dtype)
        )
        token_fraction = token_fraction / max(1, M)
        aux_loss = self.num_experts * torch.sum(token_fraction * probs.mean(dim=0))

        self._aux_loss.copy_(aux_loss.detach())
        self.routing_decisions = top_idx.detach()
        self.aux_loss_value = aux_loss.detach().item()
        self._domain_loads = domain_counts

        return out.view(B, T, C)

    # ----------------------------------------------------------- diagnostics

    def get_aux_loss(self) -> torch.Tensor:
        """Auxiliary load-balancing loss (zero if not yet computed)."""
        return self._aux_loss.clone()

    def routing_balance(self) -> float:
        """Fraction of experts that received at least one token in [0, 1]."""
        if not hasattr(self, "routing_decisions"):
            return 0.0
        unique = torch.unique(self.routing_decisions).numel()
        return unique / self.num_experts

    def domain_load(self) -> dict[str, int]:
        """Per-domain token counts from the last forward pass."""
        if not hasattr(self, "_domain_loads"):
            return {d: 0 for d in self.expert_types}
        return dict(self._domain_loads)

    def expert_load(self) -> list[int]:
        """Per-expert token counts from the last forward pass."""
        if not hasattr(self, "routing_decisions"):
            return [0] * self.num_experts
        counts = torch.zeros(self.num_experts, dtype=torch.long)
        counts.index_add_(
            0,
            self.routing_decisions.reshape(-1),
            torch.ones_like(self.routing_decisions.reshape(-1), dtype=torch.long),
        )
        return counts.tolist()
