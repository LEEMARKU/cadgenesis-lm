"""
cadgenesis.transformer.self_designing.evaluation
==========================================
Architecture evaluation for the Self-Designing Transformer.

An ``ArchitectureEvaluator`` measures a candidate ``ArchitectureSpec`` on a
real (synthetic) dataset: it builds the backbone, runs a handful of training
steps, then reports

* validation loss,
* parameter count,
* one-step latency,
* effective layer count,
* a composite **quality score** (higher is better) used by the NAS.

The composite score balances accuracy against model cost so the search does
not simply pick the largest architecture::

    quality = -val_loss * (1 - λ · log(params / params_ref)) - μ · latency_s

Complexity
----------
    evaluate(): O(S · (train_steps + 1))  where S = single training step cost
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.self_designing.architecture import ArchitectureSpec


@dataclass
class ArchitectureScore:
    """Evaluation result for a single candidate architecture."""

    spec: ArchitectureSpec
    val_loss: float
    params: int
    latency_ms: float
    effective_layers: float
    quality: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ArchitectureScore(quality={self.quality:.3f}, "
            f"val_loss={self.val_loss:.3f}, params={self.params:,}, "
            f"latency={self.latency_ms:.2f}ms, eff_layers={self.effective_layers:.1f})"
        )


class ArchitectureEvaluator:
    """
    Builds and briefly trains a candidate architecture, then scores it.

    Parameters
    ----------
    tokenizer : optional
        Not required for scoring (the model consumes raw id tensors), kept for
        interface symmetry with the training stack.
    device : str, optional
        torch device.  Defaults to CUDA if available else CPU.
    train_steps : int
        Optimizer steps used to give each candidate a fair, short head-start.
    eval_batches : int
        Number of validation batches averaged for the val-loss estimate.
    params_ref : int
        Reference parameter count for the cost penalty.
    cost_penalty : float
        λ in the quality formula above.
    latency_penalty : float
        μ in the quality formula above.
    """

    def __init__(
        self,
        tokenizer=None,
        device: str | None = None,
        train_steps: int = 20,
        eval_batches: int = 2,
        params_ref: int = 1_000_000,
        cost_penalty: float = 0.05,
        latency_penalty: float = 0.01,
    ):
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_steps = train_steps
        self.eval_batches = eval_batches
        self.params_ref = params_ref
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty

    # ------------------------------------------------------------ dataset

    def _collate(self, dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        dataset: iterable of (src_ids, tgt_ids) where tgt includes BOS..EOS.
        Returns (src, tgt_in, tgt_out) tensors for one batch.
        """
        src_list, tgt_list = [], []
        for src, tgt in dataset:
            src_list.append(torch.tensor(src, dtype=torch.long))
            tgt_list.append(torch.tensor(tgt, dtype=torch.long))
        S = max(s.shape[0] for s in src_list)
        T = max(t.shape[0] for t in tgt_list)
        pad = 0
        src = torch.stack([F.pad(s, (0, S - s.shape[0]), value=pad) for s in src_list])
        tgt = torch.stack([F.pad(t, (0, T - t.shape[0]), value=pad) for t in tgt_list])
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        return src, tgt_in, tgt_out

    def _map_type_ids(self, tgt_in: torch.Tensor, spec: ArchitectureSpec) -> torch.Tensor:
        """Coarse type map: 0 special, 1 geometry/feature, 2 numeric/other."""
        # Simple, cheap heuristic that works for any vocabulary:
        # ids < 64 are specials (type 0); the rest are typed by slot ranges.
        (64 + 1024 + 512 + 512 + 256 + 256 + 256 + 256 + 256 + 32000)
        out = torch.zeros_like(tgt_in)
        out[tgt_in < 64] = 0
        out[(tgt_in >= 64) & (tgt_in < 64 + 1024)] = 1
        out[tgt_in >= 64 + 1024] = 2
        return out

    # ------------------------------------------------------------ build/train

    def _build(self, spec: ArchitectureSpec) -> CADConfig:
        cfg = CADConfig.mini()
        cfg.model = spec.to_model_config()
        cfg.training.batch_size = 4
        cfg.training.mixed_precision = "no"
        return cfg

    def evaluate(
        self,
        spec: ArchitectureSpec,
        dataset,
    ) -> ArchitectureScore:
        """
        Train the candidate briefly and return an ArchitectureScore.

        dataset: iterable of (src_ids, tgt_ids).  ``src_ids`` is the tokenized
        language request; ``tgt_ids`` is the CAD token target (incl. BOS/EOS).
        """
        cfg = self._build(spec)
        model = GeometryAwareTransformer(cfg).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Short training head-start.
        model.train()
        try:
            for _ in range(self.train_steps):
                src, tgt_in, tgt_out = self._collate(dataset)
                src, tgt_in, tgt_out = (
                    src.to(self.device),
                    tgt_in.to(self.device),
                    tgt_out.to(self.device),
                )
                tgt_type = self._map_type_ids(tgt_in, spec).to(self.device)
                optimizer.zero_grad()
                logits, _ = model(src, tgt_in, tgt_type)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=0,
                )
                loss.backward()
                optimizer.step()
        except RuntimeError as exc:
            # Candidate too heavy for the device (OOM etc.) → worst score.
            return ArchitectureScore(
                spec=spec,
                val_loss=float("inf"),
                params=sum(p.numel() for p in model.parameters()),
                latency_ms=0.0,
                effective_layers=float(spec.num_encoder_layers + spec.num_decoder_layers),
                quality=float("-inf"),
                metadata={"error": str(exc)[:200]},
            )

        # Validation estimate + latency.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(self.eval_batches):
                src, tgt_in, tgt_out = self._collate(dataset)
                src, tgt_in, tgt_out = (
                    src.to(self.device),
                    tgt_in.to(self.device),
                    tgt_out.to(self.device),
                )
                tgt_type = self._map_type_ids(tgt_in, spec).to(self.device)
                logits, _ = model(src, tgt_in, tgt_type)
                val_loss += F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=0,
                ).item()
            latency_ms = (time.perf_counter() - start) / max(1, self.eval_batches) * 1000.0
        val_loss = val_loss / max(1, self.eval_batches)

        params = sum(p.numel() for p in model.parameters())
        effective_layers = float(spec.num_encoder_layers + spec.num_decoder_layers)
        quality = self._quality(val_loss, params, latency_ms)

        return ArchitectureScore(
            spec=spec,
            val_loss=val_loss,
            params=params,
            latency_ms=latency_ms,
            effective_layers=effective_layers,
            quality=quality,
        )

    def score(self, spec: ArchitectureSpec, dataset) -> float:
        """Convenience wrapper returning just the quality score."""
        return self.evaluate(spec, dataset).quality

    def _quality(self, val_loss: float, params: int, latency_ms: float) -> float:
        """Higher is better.  Penalises loss, parameter count and latency."""
        if not torch.isfinite(torch.tensor(val_loss)):
            return float("-inf")
        cost_ratio = max(0.0, float(params) / self.params_ref)
        return (
            -val_loss
            - self.cost_penalty * cost_ratio
            - self.latency_penalty * (latency_ms / 1000.0)
        )
