"""
cadgenesis.transformer.positional
===========================
Positional Encodings for CADGenesis-LM v2.0:
- Rotary Position Embeddings (RoPE) for 1D token sequences & 3D spatial coordinates
- Geometry Positional Encoding — additive learned encoding of 3D B-Rep spatial
  coordinates (X, Y, Z) with optional Fourier frequency features
- Attention with Linear Biases (ALiBi)
- Sinusoidal 1D Positional Encodings (legacy / fallback)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic 1D Sinusoidal Positional Encoding.

    The position table is grown on demand (doubling) when a sequence exceeds
    the initial ``max_len``, so the encoding is not a hard context-length
    limit (v6.1 §4.7 long-context support).
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self._build_table(max_len, d_model)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    @staticmethod
    def _build_table(length: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _grow(self, needed: int) -> None:
        new_len = max(needed, self.pe.shape[1] * 2)
        self.pe = self._build_table(new_len, self.d_model).unsqueeze(0)
        self.max_len = new_len

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """x: (B, T, D); ``position_offset`` shifts absolute positions (KV cache)."""
        start = position_offset
        end = position_offset + x.size(1)
        if end > self.pe.shape[1]:
            self._grow(end)
        return x + self.pe[:, start:end]


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) supporting 1D sequence position
    and 3D spatial coordinates (X, Y, Z).

    Long-context scaling (configurable, backward compatible):

    * ``scaling_type="none"`` (default) — identical to the pre-upgrade RoPE.
    * ``scaling_type="linear"`` — divides absolute positions by
      ``scaling_factor`` (YaRN/position-interpolation style).  Doubling the
      factor doubles the effective context length.
    * ``scaling_type="ntk"`` — NTK-aware rescaling: raises the base frequency
      so that high-frequency detail is preserved when extrapolating far beyond
      the trained context.
    * ``scaling_type="yarn"`` — YaRN (Yet another RoPE extensioN): interpolates
      a mix of the scaled and original frequencies with a smooth ramp over the
      low-frequency dimensions, and applies ``attn_factor = sqrt(scale)`` on
      the embeddings (equivalent to scaling attention logits).  Best
      long-context quality without fine-tuning.
    """

    # Module-level defaults (see :meth:`configure_defaults`); every attention
    # backend constructs ``RotaryEmbedding(head_dim)`` and inherits these.
    _default_max_position_embeddings: int = 4096
    _default_base: float = 10000.0
    _default_scaling_factor: float = 1.0
    _default_scaling_type: str = "none"

    @classmethod
    def configure_defaults(
        cls,
        max_position_embeddings: int | None = None,
        base: float | None = None,
        scaling_factor: float | None = None,
        scaling_type: str | None = None,
    ) -> None:
        """Set global RoPE defaults (used by all backends constructed after)."""
        if max_position_embeddings is not None:
            cls._default_max_position_embeddings = max_position_embeddings
        if base is not None:
            cls._default_base = base
        if scaling_factor is not None:
            cls._default_scaling_factor = scaling_factor
        if scaling_type is not None:
            cls._default_scaling_type = scaling_type

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int | None = None,
        base: float | None = None,
        scaling_factor: float | None = None,
        scaling_type: str | None = None,
        original_max_position_embeddings: int = 0,
    ):
        super().__init__()
        # Fall back to the module-level defaults so every backend construction
        # site (attention classes call ``RotaryEmbedding(head_dim)``) inherits
        # the model's configured long-context scaling automatically.
        max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else RotaryEmbedding._default_max_position_embeddings
        )
        base = base if base is not None else RotaryEmbedding._default_base
        scaling_factor = (
            scaling_factor
            if scaling_factor is not None
            else RotaryEmbedding._default_scaling_factor
        )
        scaling_type = (
            scaling_type if scaling_type is not None else RotaryEmbedding._default_scaling_type
        )
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be > 0.")
        if scaling_type not in ("none", "linear", "ntk", "yarn"):
            raise ValueError(
                f"scaling_type must be one of 'none', 'linear', 'ntk', 'yarn'; "
                f"got {scaling_type!r}."
            )
        if dim % 2 != 0:
            # The rotation pairs dimension i with i + dim//2 (see
            # _rotate_half), which only works for even dims.  Fail fast here
            # instead of emitting mismatched cos/sin tables downstream (the
            # MLA clamp in cad_config / geometry_transformer always picks an
            # even qk_rope_head_dim, so this is defensive).
            raise ValueError(
                f"RotaryEmbedding requires an even dim; got dim={dim}."
            )
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        self.original_max_position_embeddings = (
            original_max_position_embeddings or max_position_embeddings
        )
        self.attn_factor = math.sqrt(scaling_factor)

        arange = torch.arange(0, dim, 2).float()

        if scaling_type == "yarn":
            # YaRN: extend the base, then blend extended + original frequencies
            # with a smooth ramp over the low-frequency band.
            scale = max(scaling_factor, 1.0)
            base_ext = base * (
                (scale * self.max_position_embeddings) / self.original_max_position_embeddings
            ) ** (dim / max(dim - 2, 1))
            inv_freq_ext = 1.0 / (base_ext ** (arange / dim))
            inv_freq_orig = 1.0 / (base ** (arange / dim))
            low = math.floor(dim * math.log(scale) / (2 * math.log(base)))
            high = math.floor(dim * math.log(1 / scale) / (2 * math.log(base)))
            ramp = ((arange / dim) - low) / max(high - low, 1e-9)
            ramp = ramp.clamp(0.0, 1.0)
            inv_freq = inv_freq_ext * (1.0 - ramp) + inv_freq_orig * ramp
        else:
            effective_base = base
            if scaling_type == "ntk":
                # NTK-aware rescaling: theta' = theta * scale^(dim/(dim-2)).
                effective_base = base * (scaling_factor ** (dim / max(dim - 2, 1)))
            inv_freq = 1.0 / (effective_base ** (arange / dim))

        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cosine and sine tables (position-scaled when linear RoPE).
        t = self._scaled_positions(torch.arange(self.max_position_embeddings, dtype=torch.float))
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _scaled_positions(self, t: torch.Tensor) -> torch.Tensor:
        """Apply linear scaling to absolute positions when configured."""
        if self.scaling_type == "linear":
            return t / self.scaling_factor
        return t

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (Batch, Heads, SeqLen, HeadDim)
        position_offset: shift absolute positions (incremental KV-cache
            decoding); positions [offset, offset+SeqLen) are used.
        """
        seq_len = seq_len or q.shape[2]
        start = position_offset
        end = position_offset + seq_len
        if end > self.max_position_embeddings:
            # Dynamically extend if sequence exceeds precomputed cache
            t = self._scaled_positions(torch.arange(start, end, dtype=torch.float, device=q.device))
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(q.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[start:end].unsqueeze(0).unsqueeze(0).to(q.device)
            sin = self.sin_cached[start:end].unsqueeze(0).unsqueeze(0).to(q.device)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        # YaRN attention factor: scaling both q and k by 1/sqrt(attn_factor)
        # is equivalent to dividing the attention logits by ``attn_factor``.
        if self.scaling_type == "yarn":
            q_embed = q_embed / self.attn_factor
            k_embed = k_embed / self.attn_factor
        return q_embed, k_embed


class ALiBiBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) penalty matrix generator.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(self._get_slopes(num_heads)).unsqueeze(1).unsqueeze(2)  # (H, 1, 1)
        self.slopes: torch.Tensor
        self.register_buffer("slopes", slopes, persistent=False)

    def _get_slopes(self, n: int) -> list[float]:
        def get_slopes_power_of_2(n_p2: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n_p2) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n_p2)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_pow2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_pow2)
                + self._get_slopes(2 * closest_pow2)[0::2][: n - closest_pow2]
            )

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns bias matrix of shape (1, num_heads, seq_len, seq_len)
        """
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0)  # (1, seq_len, seq_len)
        alibi = -1.0 * self.slopes.to(device) * relative_position.unsqueeze(0)
        return alibi


class GeometryPositionalEncoding(nn.Module):
    """
    Geometry Positional Encoding — additive, learned encoding of 3D spatial
    coordinates into the model dimension.

    Rationale
    ---------
    1D sinusoidal / RoPE encodings only encode *token order*; they carry no
    notion of the metric coordinates of B-Rep vertices, sketch points, or
    feature origins.  This module embeds per-token (X, Y, Z) coordinates so
    the attention mixture can exploit *metric* locality (e.g. two sketches
    close in space) in addition to token order.

    Design
    ------
    * ``use_fourier=True`` (default): coordinates are first expanded into
      Fourier features ``{sin(2π · 2^i · x_d), cos(2π · 2^i · x_d)}`` for
      ``i in [0, num_frequencies)`` and each dimension ``d ∈ {X, Y, Z}`` —
      a standard positional-encoding technique that lets the linear
      projection carve high-frequency spatial detail from the low-frequency
      bulk.  When ``use_fourier=False`` the raw coordinates are used.
    * A learned ``nn.Linear`` projects the features into ``d_model``.
    * A learned scalar ``scale`` modulates the overall contribution
      (initialised to 1) so the encoding is gradient-tuneable.
    * ``forward(x, coords)`` returns ``x + enc`` (additive, residual-safe);
      ``embed(coords)`` returns the raw encoding.

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    use_fourier : bool
        Whether to expand coordinates with Fourier frequency features.
    num_frequencies : int
        Number of frequency octaves per coordinate axis (default 8).
    scale_init : float
        Initial value of the learned scale (default 1.0).
    """

    _COORD_DIMS = 3  # X, Y, Z

    def __init__(
        self,
        d_model: int,
        use_fourier: bool = True,
        num_frequencies: int = 8,
        scale_init: float = 1.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1.")

        self.d_model = d_model
        self.use_fourier = use_fourier
        self.num_frequencies = num_frequencies

        in_features = self._COORD_DIMS * 2 * num_frequencies if use_fourier else self._COORD_DIMS
        self.coord_proj = nn.Linear(in_features, d_model)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, T, 3) or (T, 3) → encoding (B, T, d_model) or (T, d_model)
        """
        if coords.dim() == 2:
            features = self._features(coords)  # (T, in_features)
        elif coords.dim() == 3:
            features = self._features(coords)  # (B, T, in_features)
        else:
            raise ValueError(
                f"coords must be (T, 3) or (B, T, 3); got shape {tuple(coords.shape)}."
            )
        if features.shape[-1] != self.coord_proj.in_features:
            raise ValueError(f"coords last dim must be 3 (XYZ); got {coords.shape[-1]}.")
        return self.coord_proj(features) * self.scale

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        """Expand coordinates into the raw feature vector."""
        if not self.use_fourier:
            return coords
        # (…, 3, 2*num_frequencies): per-axis sin/cos at each octave.
        freq = 2.0 ** torch.arange(
            self.num_frequencies, dtype=coords.dtype, device=coords.device
        )  # (num_frequencies,)
        angles = coords.unsqueeze(-1) * freq.reshape(1, 1, -1)  # (B, T, 3, F)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        coords: (B, T, 3) or (T, 3)
        Returns: x + geometry positional encoding (same shape as x).
        If coords is None, returns x unchanged (no-op).
        """
        if coords is None:
            return x
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(x.shape[0], -1, -1)
        enc = self.embed(coords)
        if enc.dim() == 2:
            enc = enc.unsqueeze(0)
        if enc.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"coord batch/time {tuple(enc.shape[:2])} must match input {tuple(x.shape[:2])}."
            )
        return x + enc
