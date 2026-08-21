"""cadgenesis.transformer.heads
=============================
Output heads for CADGenesis-LM v6.0: language-modeling head and confidence
head, with optional weight tying.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Linear language-modeling head mapping hidden states to vocabulary logits.

    Supports weight tying with an input token embedding via ``tie_weights``.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: nn.Embedding | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1 or vocab_size < 1:
            raise ValueError("d_model and vocab_size must be positive")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.out_proj = nn.Linear(d_model, vocab_size, bias=bias)
        if tie_weights is not None:
            if tie_weights.weight.shape[0] != vocab_size:
                raise ValueError(
                    f"cannot tie weights: embedding vocab {tie_weights.weight.shape[0]} "
                    f"!= LMHead vocab {vocab_size}"
                )
            self.out_proj.weight = tie_weights.weight
        self._tied = tie_weights is not None

    @property
    def is_tied(self) -> bool:
        return self._tied

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (..., d_model) -> (..., vocab_size)"""
        return self.out_proj(hidden_states)


class ConfidenceHead(nn.Module):
    """Maps hidden states to a scalar confidence logit per token."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (..., d_model) -> (..., 1)"""
        return self.out_proj(hidden_states)


class OutputHeads(nn.Module):
    """Convenience container bundling the LM head and the confidence head."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.lm_head = LMHead(d_model, vocab_size, tie_weights=tie_weights)
        self.confidence_head = ConfidenceHead(d_model)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits (..., vocab_size), confidence_logits (..., 1))``."""
        return self.lm_head(hidden_states), self.confidence_head(hidden_states)
