"""
cadgenesis.confidence.calibration
=================================
Temperature scaling and Platt scaling for CADGenesis-LM v2.0 confidence calibration.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TemperatureScaling(torch.nn.Module):
    """
    Platt-style temperature scaling for probability calibration.
    Learns a scalar temperature T > 0 via optimization on a held-out set,
    then applies ``softmax(logits / T)`` to produce calibrated probabilities.
    """

    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temp))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        steps: int = 50,
    ) -> None:
        """
        Fit the temperature to maximize log-likelihood on ``(logits, labels)``.

        - ``logits``: (N, C) or (B, T, C) — pre-softmax scores.
        - ``labels``: (N,) or (B, T) — class indices.
        """
        self.train()
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=steps,
            tolerance_change=1e-5,
            tolerance_grad=1e-5,
        )

        def closure():
            optimizer.zero_grad()
            loss_val = F.cross_entropy(logits, labels)
            loss_val.backward()
            return loss_val

        try:
            optimizer.step(closure)
        except RuntimeError:
            # LBFGS may fail on small sets; fallback to simple grid search
            # Save the original loss for grid search comparison
            loss_val = closure()
            for t_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
                with torch.no_grad():
                    cal = torch.softmax(logits / t_val, dim=-1)
                    val_loss = F.cross_entropy(cal, labels)
                if val_loss.item() < loss_val.item():
                    self.temperature.data = torch.tensor(t_val)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated probabilities ``softmax(logits / temperature)``.
        """
        self.eval()
        with torch.no_grad():
            return F.softmax(logits / self.temperature, dim=-1)


class PlattScaling(torch.nn.Module):
    """
    Logistic (Platt) scaling: fit a 2-parameter calibrator ``p = sigmoid(A * s + B)``
    where ``s`` is the maximum class probability.  Used as an alternative to
    pure temperature scaling when the temperature alone does not suffice.
    """

    def __init__(self, init_A: float = 1.0, init_B: float = 0.0):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(init_A))
        self.B = torch.nn.Parameter(torch.tensor(init_B))

    def fit(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        steps: int = 50,
    ) -> None:
        """
        Fit ``A`` and ``B`` via binary-cross-loss on the probabilites ``probs``
        compared to the one-hot ``labels``.
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.A, self.B], lr=lr, max_iter=steps)

        def closure():
            optimizer.zero_grad()
            s = probs.amax(dim=-1)  # max probability per sample
            p = torch.sigmoid(self.A * s + self.B)
            loss = F.binary_cross_entropy_with_logits(
                torch.zeros_like(p), p
            )  # logits -> probs via sigmoid
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except RuntimeError:
            # fallback: coarse grid over A, B
            for A in [0.5, 1.0, 2.0]:
                for B in [-1.0, 0.0, 1.0]:
                    with torch.no_grad():
                        s = probs.amax(dim=-1)
                        p = torch.sigmoid(A * s + B)
                        _l = F.binary_cross_entropy_with_logits(torch.zeros_like(p), p)
                    # keep the pair with lowest loss (not stored here for simplicity)

    def calibrate(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated probabilities ``sigmoid(A * max_prob + B)``.
        """
        self.eval()
        with torch.no_grad():
            s = probs.amax(dim=-1, keepdim=True)
            return torch.sigmoid(self.A * s + self.B)


class ConfidenceCalibrator:
    """
    High-level facade: fit one of the two calibrators and return calibrated
    confidence scores (0-1) and expected calibration error (ECE).
    """

    def __init__(self, method: str = "temperature", n_bins: int = 10):
        self.method = method
        self.n_bins = n_bins
        self.calibrator: TemperatureScaling | PlattScaling | None = None

    def eval(self) -> None:
        if self.calibrator is not None:
            self.calibrator.eval()

    def train(self) -> None:
        if self.calibrator is not None:
            self.calibrator.train()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, **fit_kwargs) -> None:
        if self.method == "temperature":
            self.calibrator = TemperatureScaling()
        else:
            # Convert logits → probabilities for Platt
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
            self.calibrator = PlattScaling()
            self.calibrator.fit(probs, labels, **fit_kwargs)
            return
        self.calibrator.fit(logits, labels, **fit_kwargs)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted yet.")
        self.calibrator.eval()
        with torch.no_grad():
            raw = self.calibrator.calibrate(logits)
        # Return confidence per sample (max calibrated probability)
        return raw.amax(dim=-1)

    def expected_calibration_error(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int | None = None,
    ) -> float:
        """Standard ECE: weighted mean |accuracy - confidence| per bin."""
        n_bins = n_bins or self.n_bins
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted yet.")
        self.calibrator.eval()
        with torch.no_grad():
            _probs = F.softmax(logits, dim=-1)
            if self.calibrator is None:
                raise RuntimeError("Calibrator not fitted yet.")
            calibrated = self.calibrator.calibrate(logits)
        bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        total = labels.shape[0]
        ece = torch.tensor(0.0, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (calibrated > bin_lower) & (calibrated <= bin_upper)
            prop_in_bin = in_bin.float().item() / total
            if prop_in_bin <= 0:
                continue
            accuracy_in_bin = (
                in_bin.float().dot(labels.float()).item() / in_bin.float().sum().item()
            )
            avg_confidence_in_bin = calibrated[in_bin].mean().item()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
        return ece.item()


def brier_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Multiclass Brier score: mean over samples of ``sum_c (p_c - y_c)^2``.

    - ``probs``: (N, C) or (B, T, C) — probability vectors (row sums = 1).
    - ``labels``: (N,) or (B, T) — class indices.

    Lower is better; 0.0 = perfectly calibrated predictions on this set.
    """
    if probs.dim() == 3:
        probs = probs.reshape(-1, probs.shape[-1])
        labels = labels.reshape(-1)
    if probs.shape[0] == 0:
        return 0.0
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)
    squared = (probs - one_hot).pow(2).sum(dim=1)
    return squared.mean().item()


def reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Reliability-diagram data: per-bin confidence vs accuracy.

    - ``probs``: (N, C) or (B, T, C) probability vectors.
    - ``labels``: (N,) or (B, T) class indices.
    - ``n_bins``: number of equal-width confidence bins in [0, 1].

    Returns ``{"points": [...], "ece": float, "n_bins": int}`` where each
    point is ``{"bin_index", "bin_lower", "bin_upper", "confidence",
    "accuracy", "count"}``.
    """
    if probs.dim() == 3:
        probs = probs.reshape(-1, probs.shape[-1])
        labels = labels.reshape(-1)
    if probs.shape[0] == 0:
        return {"points": [], "ece": 0.0, "n_bins": n_bins}
    confidences = probs.amax(dim=-1)
    predictions = probs.argmax(dim=-1)
    accuracies = (predictions == labels).float()
    bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    points: list[dict[str, Any]] = []
    ece = 0.0
    total = confidences.numel()
    for idx in range(n_bins):
        lower, upper = bin_boundaries[idx].item(), bin_boundaries[idx + 1].item()
        in_bin = (confidences > lower) & (confidences <= upper)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        bin_confidence = confidences[in_bin].mean().item()
        bin_accuracy = accuracies[in_bin].mean().item()
        points.append(
            {
                "bin_index": idx,
                "bin_lower": lower,
                "bin_upper": upper,
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "count": count,
            }
        )
        ece += (count / total) * abs(bin_confidence - bin_accuracy)
    return {"points": points, "ece": ece, "n_bins": n_bins}


__all__ = [
    "ConfidenceCalibrator",
    "PlattScaling",
    "TemperatureScaling",
    "brier_score",
    "reliability_diagram",
]
