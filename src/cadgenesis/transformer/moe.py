"""
cadgenesis.transformer.moe
====================
Sparse Mixture-of-Experts FFN (MoE) primitive for CADGenesis-LM v2.0.

Purpose
-------
Replaces the dense SwiGLU feed-forward network in selected transformer blocks
with a sparse, token-routed mixture of experts.  Each token is dispatched to
the ``top_k`` most relevant experts by a learned router.  The expert count is
*expandable* at runtime, which is the substrate for the "Sparse Expert Growth"
capability of the Self-Designing Transformer (see
``cadgenesis.transformer.self_designing``).

Architecture
------------
::

    SparseMoEFFN
    ├── router      : nn.Linear(d_model, num_experts)  (softmax over experts)
    ├── experts     : ModuleList[_MoEExpert]           (growable)
    └── forward     : top-k dispatch + weighted recombination + aux loss

Algorithms
----------
    top-k routing:
        router_logits = Router(x)                         (M, E)
        probs         = softmax(router_logits)            (M, E)
        topk_probs, topk_idx = probs.topk(k=top_k)        (M, k)
        weights       = topk_probs / sum(topk_probs)      normalized
        out[x_i]      = Σ_k weights[i,k] · expert_{idx[i,k]}(x_i)

    Auxiliary load-balancing loss (Shazeer et al. 2017, Switch Transformers):
        f_e = fraction of tokens routed to expert e
        P_e = mean router probability assigned to expert e
        aux_loss = E · Σ_e f_e · P_e

Complexity
----------
    Time:   O(M · (k · E · d²))  where M = tokens, E = experts, k = top_k
    Space:  O(M · k)             routing decisions
    Growth: add_expert() is O(E) amortized (router re-projection is O(E·d))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MoEExpert(nn.Module):
    """A single expert: a two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M, d_model) → (M, d_model)"""
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class SparseMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts Feed-Forward Network with top-k routing.

    The expert pool is growable via :meth:`add_expert`, enabling online
    "sparse expert growth".  An auxiliary load-balancing loss is exposed
    through :meth:`get_aux_loss` so a trainer can mix it into the total loss.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    num_experts : int
        Initial number of experts (>= ``top_k``).
    top_k : int
        Number of experts activated per token.
    expert_dim : int, optional
        Hidden width of each expert.  Defaults to ``2 * d_model``.
    dropout : float
        Dropout probability inside each expert.
    router_jitter : float
        Uniform noise magnitude added to router logits during training.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_dim: int | None = None,
        dropout: float = 0.1,
        router_jitter: float = 0.02,
        use_aux_free_balancing: bool = False,
        balance_speed: float = 0.001,
        z_loss_weight: float = 1e-3,
        capacity_factor: float | None = None,
        drop_tokens: bool = False,
        num_shared_experts: int = 0,
        shared_expert_dim: int | None = None,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts.")
        if capacity_factor is not None and capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0.")
        if drop_tokens and capacity_factor is None:
            raise ValueError("capacity_factor is required when drop_tokens=True.")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0.")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim or (2 * d_model)
        self.router_jitter = router_jitter
        self.use_aux_free_balancing = use_aux_free_balancing
        self.balance_speed = balance_speed
        self.z_loss_weight = z_loss_weight
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        self.router = nn.Linear(d_model, num_experts, bias=False)
        # DeepSeek-V3 style per-expert bias (aux-loss-free balancing).
        self.expert_bias: torch.Tensor = nn.Parameter(torch.zeros(num_experts))
        self.experts: nn.ModuleList = nn.ModuleList(
            [_MoEExpert(d_model, self.expert_dim, dropout) for _ in range(num_experts)]
        )

        # DeepSeek-V3 shared expert: one always-on expert fused into every
        # token's output *in addition* to the top-k routed experts.  Concentrates
        # common knowledge and lets the routed experts stay small/fine-grained.
        self.num_shared_experts = num_shared_experts
        self.shared_expert = (
            _MoEExpert(d_model, shared_expert_dim or (d_model // 2), dropout)
            if num_shared_experts > 0
            else None
        )

        # Auxiliary loss accumulator (reset at the start of each forward pass).
        self._aux_loss: torch.Tensor
        self.register_buffer("_aux_loss", torch.zeros(()), persistent=False)
        self._aux_loss.zero_()

    # ------------------------------------------------------------------ growth

    def add_expert(self, expert: nn.Module | None = None) -> int:
        """
        Append a new expert and expand the router.

        Existing router rows are copied verbatim and the new row is initialised
        with small normal noise so already-learned routing behaviour is
        preserved while the new expert starts with an exploratory prior.
        """
        expert = expert or _MoEExpert(self.d_model, self.expert_dim)
        self.experts.append(expert)
        self.num_experts += 1

        new_router = nn.Linear(self.d_model, self.num_experts, bias=False)
        with torch.no_grad():
            new_router.weight[: self.num_experts - 1].copy_(self.router.weight)
            new_router.weight[self.num_experts - 1 :].normal_(0.0, 0.02)
        self.router = new_router
        with torch.no_grad():
            new_bias = torch.zeros(self.num_experts)
            new_bias[: self.num_experts - 1].copy_(self.expert_bias)
            self.expert_bias = nn.Parameter(new_bias)
        return self.num_experts

    def remove_expert(self, index: int) -> int:
        """
        Retire an expert by index (zero-based).  Used by the adapter/evolution
        lifecycle to shrink under-utilised experts.  The router column and
        expert bias entry are dropped as well.
        """
        if not (0 <= index < self.num_experts):
            raise IndexError(f"Expert index {index} out of range.")
        del self.experts[index]
        self.num_experts -= 1

        kept = [i for i in range(self.num_experts + 1) if i != index]
        new_router = nn.Linear(self.d_model, self.num_experts, bias=False)
        with torch.no_grad():
            for new_i, old_i in enumerate(kept):
                new_router.weight[new_i].copy_(self.router.weight[old_i])
        self.router = new_router
        with torch.no_grad():
            new_bias = torch.zeros(self.num_experts)
            for new_i, old_i in enumerate(kept):
                new_bias[new_i] = self.expert_bias[old_i]
            self.expert_bias = nn.Parameter(new_bias)
        return self.num_experts

    # ------------------------------------------------------------- routing/FFN

    def _router_probs(self, flat: torch.Tensor) -> torch.Tensor:
        """softmax router probabilities with optional train-time jitter."""
        logits = self.router(flat)
        if self.training and self.router_jitter > 0.0:
            logits = logits + torch.empty_like(logits).uniform_(
                -self.router_jitter, self.router_jitter
            )
        return F.softmax(logits, dim=-1)

    def _aux_free_balance_step(self, probs: torch.Tensor) -> None:
        """
        DeepSeek-V3 style expert-bias update (no auxiliary loss): bias_shift
        is applied to the *router logits* (pre-softmax), so this update runs
        on the logits-space bias in-place, without gradients.
        """
        with torch.no_grad():
            mean_probs = probs.mean(dim=0)  # (E,)
            self.expert_bias -= self.balance_speed * torch.sign(mean_probs - mean_probs.mean())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) → (B, T, d_model)

        Stores the per-token routing decisions on ``self.routing_decisions``
        and the auxiliary load-balancing loss on ``self.aux_loss_value`` for
        diagnostics / training.

        When ``use_aux_free_balancing=True`` the DeepSeek-V3 mechanism is
        used instead: a learned per-expert bias (shifted in-place during
        training) plus an optional router z-loss — no auxiliary load-balancing
        loss.  When ``drop_tokens=True`` (requires ``capacity_factor``),
        tokens beyond each expert's capacity are zeroed out (expert dropout).
        """
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        M = flat.shape[0]

        logits = self.router(flat) + self.expert_bias  # (M, E)
        if self.training and self.router_jitter > 0.0:
            logits = logits + torch.empty_like(logits).uniform_(
                -self.router_jitter, self.router_jitter
            )
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (M, k)
        norm = top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = top_probs / norm  # (M, k)

        out = torch.zeros_like(flat)

        if self.use_aux_free_balancing:
            # Router z-loss: mean logsumexp^2 (regularises logits magnitude).
            z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
            self.z_loss_value = float((self.z_loss_weight * z_loss).item())
            # Keep the *live* tensor (not detached) so the trainer can mix it
            # into the loss; gradients flow through the router logits.
            self._aux_loss.copy_(self.z_loss_weight * z_loss)
            if self.training:
                self._aux_free_balance_step(probs)
        else:
            self.z_loss_value = 0.0
            # Token counts per expert → auxiliary load-balancing loss.  Each
            # token is counted exactly once (via its top-1 expert), matching
            # the Switch-Transformer formulation (Shazeer et al., 2022).  This
            # loop only *counts*; the actual routed-expert computation happens
            # once, in the capacity-aware loop below (which also handles token
            # dropping).
            token_fraction = torch.zeros(self.num_experts, device=x.device)
            selected = top_idx[:, 0]
            for e in range(self.num_experts):
                token_fraction[e] = (selected == e).float().sum()
            token_fraction = token_fraction / max(1, M)
            aux_loss = self.num_experts * torch.sum(token_fraction * probs.mean(dim=0))
            # Keep the live tensor: gradients flow through probs.mean(dim=0).
            self._aux_loss.copy_(aux_loss)

        # Expert capacity + token dropping (train-time only, DeepSeek-V3).
        capacity = None
        if self.capacity_factor is not None:
            capacity = max(1, math.ceil(self.capacity_factor * M / self.num_experts))
        for k in range(self.top_k):
            selected = top_idx[:, k]
            w_k = weights[:, k]
            for e in range(self.num_experts):
                mask = selected == e
                if not mask.any():
                    continue
                if self.drop_tokens and capacity is not None and mask.sum() > capacity:
                    w = w_k[mask]
                    keep = w.topk(capacity).indices
                    keep_set = torch.zeros(int(mask.sum()), dtype=torch.bool, device=x.device)
                    keep_set[keep] = True
                    m_flat = torch.nonzero(mask, as_tuple=False).flatten()
                    out[m_flat[keep_set]] = out[m_flat[keep_set]] + (
                        w[keep_set].unsqueeze(-1) * self.experts[e](flat[m_flat[keep_set]])
                    )
                else:
                    out[mask] = out[mask] + w_k[mask].unsqueeze(-1) * self.experts[e](flat[mask])

        self.routing_decisions = top_idx.detach()
        self.aux_loss_value = self._aux_loss.item()

        # DeepSeek-V3 shared expert: always active, added to every token.
        if self.shared_expert is not None:
            out = out + self.shared_expert(flat)

        return out.view(B, T, C)

    def get_aux_loss(self) -> torch.Tensor:
        """Return the auxiliary load-balancing loss (zero if not yet computed)."""
        return self._aux_loss.clone()

    def routing_balance(self) -> float:
        """
        Fraction of experts that received at least one token in the last
        forward pass — a coarse measure of routing health in [0, 1].
        """
        if not hasattr(self, "routing_decisions"):
            return 0.0
        unique = torch.unique(self.routing_decisions).numel()
        return unique / self.num_experts

    def expert_load(self) -> list[int]:
        """Per-expert token counts from the last forward pass (list of ints)."""
        if not hasattr(self, "routing_decisions"):
            return [0] * self.num_experts
        counts = torch.zeros(self.num_experts, dtype=torch.long)
        counts.index_add_(
            0,
            self.routing_decisions.reshape(-1),
            torch.ones_like(self.routing_decisions.reshape(-1), dtype=torch.long),
        )
        return counts.tolist()
