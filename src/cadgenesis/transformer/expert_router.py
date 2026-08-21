"""cadgenesis.transformer.expert_router
=====================================
Standalone top-k expert router for the sparse mixture of experts.

This is the canonical implementation of the routing decision used by
:class:`cadgenesis.transformer.moe.SparseMoEFFN`.  It is exposed as its own
module so the routing logic can be reused, tested and evolved in isolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    """
    Top-k softmax expert router with optional train-time jitter.

    Dispatches each token to the ``top_k`` most probable experts and returns
    the normalised combination weights alongside the chosen indices.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    num_experts : int
        Number of experts to route across.
    top_k : int
        Number of experts activated per token; must satisfy
        ``1 <= top_k <= num_experts``.
    jitter : float
        Uniform noise magnitude added to router logits during training
        (Switch-Transformer style noise injection).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        jitter: float = 0.02,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.linear = nn.Linear(d_model, num_experts, bias=False)

    def grow(self, n: int = 1) -> None:
        """Expand the router to ``num_experts + n`` experts, preserving old rows."""
        if n < 1:
            raise ValueError("n must be >= 1")
        old = self.linear
        self.num_experts += n
        new = nn.Linear(self.d_model, self.num_experts, bias=False)
        with torch.no_grad():
            new.weight[: old.weight.shape[0]].copy_(old.weight)
            new.weight[old.weight.shape[0] :].normal_(0.0, 0.02)
        self.linear = new

    def forward(
        self,
        flat: torch.Tensor,
        use_jitter: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flat: Token vectors of shape ``(M, d_model)``.
            use_jitter: Override train-time jitter (defaults to ``self.training``).

        Returns:
            ``(weights (M, top_k), top_idx (M, top_k))`` with weights already
            normalised across the selected experts.
        """
        if use_jitter is None:
            use_jitter = self.training
        logits = self.linear(flat)
        if use_jitter and self.jitter > 0.0:
            logits = logits + torch.empty_like(logits).uniform_(-self.jitter, self.jitter)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        norm = top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return top_probs / norm, top_idx

    def load_balance_loss(self, probs: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """
        Shazeer et al. auxiliary load-balancing loss.

        ``probs``: router softmax probabilities (M, E) from the last forward.
        ``top_idx``: chosen expert indices (M, top_k).
        Returns a scalar loss promoting uniform token-to-expert distribution.
        """
        M = probs.shape[0]
        if M == 0:
            return torch.tensor(0.0, device=probs.device)
        token_fraction = torch.zeros(self.num_experts, device=probs.device)
        token_fraction.index_add_(
            0, top_idx.reshape(-1), torch.ones_like(top_idx.reshape(-1), dtype=probs.dtype)
        )
        token_fraction = token_fraction / M
        return self.num_experts * torch.sum(token_fraction * probs.mean(dim=0))
