"""cadgenesis.confidence.monitoring
================================
Confidence distribution monitoring.

Provides tools for monitoring and analyzing the distribution of
confidence scores across CADGenesis-LM predictions, enabling
drift detection, out-of-distribution detection, and confidence
calibration analysis.
"""

from __future__ import annotations

from typing import Any

import torch


class ConfidenceMonitor:
    """Monitor confidence score distributions over time.

    Tracks statistics such as mean, median, percentiles, and
    drift between batches or time windows.
    """

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.confidences: list[float] = []
        self.step = 0

    def update(self, confidences: list[float] | torch.Tensor) -> None:
        """Update monitor with new batch of confidence scores.

        - ``confidences``: 1-D list or tensor of confidence values in [0, 1].
        """
        if isinstance(confidences, torch.Tensor):
            self.confidences.extend(confidences.tolist())
        else:
            self.confidences.extend(confidences)
        # Truncate history if exceeding max
        if len(self.confidences) > self.max_history:
            self.confidences = self.confidences[-self.max_history :]
        self.step += 1

    def summary(self) -> dict[str, float]:
        """Return summary statistics of observed confidences.

        Returns dict with keys
            - ``"mean"``: mean confidence
            - ``"median"``: median confidence
            - ``"p10"``: 10th percentile
            - ``"p90"``: 90th percentile
            - ``"std"``: standard deviation
            - ``"count"``: number of observations
        """
        if not self.confidences:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "p10": float("nan"),
                "p90": float("nan"),
                "std": float("nan"),
                "count": 0,
            }

        c = torch.tensor(self.confidences)
        return {
            "mean": float(c.mean().item()),
            "median": float(torch.median(c).item()),
            "p10": float(torch.kthvalue(c, int(0.1 * len(c) + 1)).values.item()),
            "p90": float(torch.kthvalue(c, int(0.9 * len(c) + 1)).values.item()),
            "std": float(c.std(unbiased=False).item()),
            "count": len(self.confidences),
        }


class DistributionDriftDetector:
    """Detect drift in confidence distribution between reference and current samples.

    Uses Kolmogorov-Smirnov test approximation via percentile comparison.
    """

    def __init__(self, reference: torch.Tensor | None = None, threshold: float = 0.05):
        self.reference = reference
        self.threshold = threshold

    def update_reference(self, reference: torch.Tensor) -> None:
        """Set the reference confidence distribution.

        - ``reference``: 1-D tensor of reference confidence values.
        """
        self.reference = reference

    def detect(self, current: torch.Tensor) -> dict[str, Any]:
        """Detect drift between reference and current confidence distributions.

        Returns dict with:
            - ``"drift_detected"``: Whether drift exceeds threshold
            - ``"ks_statistic"``: Approximate KS statistic
            - ``"p_value"``: Approximate p-value
            - ``"mean_shift"``: Difference in means
            - ``"p90_shift"``: Difference in 90th percentile
        """
        if self.reference is None or self.reference.numel() == 0:
            return {
                "drift_detected": False,
                "ks_statistic": float("nan"),
                "p_value": float("nan"),
                "mean_shift": float("nan"),
                "p90_shift": float("nan"),
            }

        ref = self.reference
        cur = current

        # Compute means
        mean_ref = ref.float().mean().item()
        mean_cur = cur.float().mean().item()
        mean_shift = mean_cur - mean_ref

        # Compute p90 shifts
        p90_ref = torch.kthvalue(ref, int(0.9 * len(ref) + 1)).values.item()
        p90_cur = torch.kthvalue(cur, int(0.9 * len(cur) + 1)).values.item()
        p90_shift = p90_cur - p90_ref

        # Simple KS approximation via percentile difference
        ks_stat = abs(p90_shift)  # proxy
        drift_detected = abs(ks_stat) > self.threshold

        return {
            "drift_detected": drift_detected,
            "ks_statistic": ks_stat,
            "p_value": float("nan"),  # full KS test would require scipy
            "mean_shift": mean_shift,
            "p90_shift": p90_shift,
        }
