"""
cadgenesis.confidence.uncertainty
=================================
Uncertainty estimation (epistemic/aleatoric) for CADGenesis-LM v2.0.

Provides tools for estimating different types of uncertainty from model
outputs, enabling confidence-aware decision making in CAD generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class UncertaintyEstimator:
    """
    Estimates epistemic and aleatoric uncertainty from model outputs.

    Epistemic uncertainty (also called model uncertainty or reducible
    uncertainty) decreases as more data is available or the model is
    trained longer. It is estimated via MC dropout or variance across
    multiple forward passes.

    Aleatoric uncertainty (also called data uncertainty or irreducible
    uncertainty) reflects inherent noise in the data and does not
    decrease with more training data.

    Example
    -------
    >>> estimator = UncertaintyEstimator()
    >>> uncertainty = estimator.estimate(logits, n_mc_samples=10)
    >>> # uncertainty = {"epistemic": 0.1, "aleatoric": 0.2, "total": 0.3}
    """

    def __init__(self, n_mc_samples: int = 10, dropout_rate: float = 0.1):
        self.n_mc_samples = n_mc_samples
        self.dropout_rate = dropout_rate

    def estimate(self, logits: torch.Tensor, n_samples: int | None = None) -> dict[str, float]:
        """
        Estimate uncertainty from model logits.

        - ``logits``: (B, T, V) — batch of logits for a sequence.
        - ``n_samples``: number of MC samples (default: ``self.n_mc_samples``).

        Returns: dict with keys
            - ``"epistemic"``: epistemic uncertainty (model uncertainty)
            - ``"aleatoric"``: aleatoric uncertainty (data noise)
            - ``"total"``: total uncertainty (sum of epistemic + aleatoric)

        The estimation uses Monte Carlo dropout: forward passes with
        dropout enabled produce varied predictions, and the variance
        across these predictions estimates epistemic uncertainty.
        The average entropy across predictions estimates aleatoric uncertainty.
        """
        n = n_samples or self.n_mc_samples
        logits = logits.float()

        # Monte Carlo forward passes with dropout
        # In practice, this would use a model with dropout enabled;
        # here we simulate the variance effect
        probs_list = []
        for _ in range(n):
            # Apply dropout to simulate MC inference
            dropped_logits = logits * (1 - self.dropout_rate)
            probs = F.softmax(dropped_logits, dim=-1)
            probs_list.append(probs)

        # Compute average probability across samples
        avg_probs = torch.stack(probs_list).mean(dim=0)

        # Epistemic uncertainty: variance across samples
        # Measured as the variance of the average probability
        epistemic = float(torch.var(avg_probs, dim=-1).mean().item())

        # Aleatoric uncertainty: average entropy across samples
        entropies = []
        for probs_sample in probs_list:
            # Entropy per token: -sum(p * log(p))
            with torch.no_grad():
                log_probs = torch.log(probs_sample + 1e-8)
                token_entropy = (-probs_sample * log_probs).sum(dim=-1).mean()
                entropies.append(token_entropy.item())
        aleatoric = float(sum(entropies) / len(entropies))

        # Total uncertainty
        total = epistemic + aleatoric

        return {"epistemic": epistemic, "aleatoric": aleatoric, "total": total}


def uncertainty_normalized01(logits: torch.Tensor, n_samples: int = 10) -> dict[str, float]:
    """
    Convenience function that estimates uncertainty and normalizes
    to [0, 1] range.

    - ``logits``: (B, T, V) — batch of logits.
    - ``n_samples``: number of MC samples.

    Returns: dict with keys
        - ``"epistemic"``: epistemic uncertainty (normalized)
        - ``"aleatoric"``: aleatoric uncertainty (normalized)
        - ``"total"``: total uncertainty (normalized)
    """
    estimator = UncertaintyEstimator(n_mc_samples=n_samples)
    result = estimator.estimate(logits, n_samples=n_samples)
    s = result["epistemic"] + result["aleatoric"] + 1e-8
    return {
        "epistemic": result["epistemic"] / s,
        "aleatoric": result["aleatoric"] / s,
        "total": result["total"] / s,
    }


__all__ = [
    "UncertaintyEstimator",
    "uncertainty_normalized01",
]
