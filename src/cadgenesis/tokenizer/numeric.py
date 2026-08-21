"""
cadgenesis.tokenizer.numeric
=============================
Numeric parameter token family.

Purpose
-------
Maps continuous CAD parameters (lengths, angles, radii, offsets, …) to
discrete integer tokens.  All numeric domain tokenizers route through this
module so the quantization scheme is consistent across the entire system.

Architecture
------------
Two quantization strategies are provided:

1. **Uniform bins** (default) — equal-width partitioning of [param_min, param_max].
   Fast and deterministic; good for training from scratch.

2. **Log-uniform bins** — useful when parameters span multiple orders of
   magnitude (e.g. 0.01 mm tolerances alongside 500 mm structural dimensions).

For each bin ``i`` the corresponding token string is ``NUM_i`` (where ``i``
is a zero-padded integer to keep token strings lexicographically sortable).

Angle parameters use a separate set: ``ANG_i`` with 360 bins (1° resolution).

Algorithms
----------
    encode_value(v)  → token_str       O(log B) binary-search
    decode_token(s)  → float           O(1) array lookup

Complexity
----------
    Space:  O(B)  where B = num_bins (default 256)
    Encode: O(log B)
    Decode: O(1)
"""

from __future__ import annotations

import bisect
import math

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Bin boundary computation
# ---------------------------------------------------------------------------


def _uniform_bins(lo: float, hi: float, n: int) -> tuple[list[float], list[float]]:
    """
    Returns n+1 boundary edges and n bin centres for the range [lo, hi].

    The bin centre for bin i is the midpoint of [edges[i], edges[i+1]).
    """
    step = (hi - lo) / n
    edges = [lo + i * step for i in range(n + 1)]
    centres = [(edges[i] + edges[i + 1]) / 2.0 for i in range(n)]
    return edges, centres


def _log_uniform_bins(lo: float, hi: float, n: int) -> tuple[list[float], list[float]]:
    """
    Returns n+1 boundary edges and n bin centres using log-uniform spacing.
    Requires lo > 0.
    """
    if lo <= 0:
        raise ValueError("log-uniform bins require lo > 0.")
    log_lo, log_hi = math.log(lo), math.log(hi)
    step = (log_hi - log_lo) / n
    edges = [math.exp(log_lo + i * step) for i in range(n + 1)]
    centres = [(edges[i] + edges[i + 1]) / 2.0 for i in range(n)]
    return edges, centres


# ---------------------------------------------------------------------------
# NumericQuantizer — the core encode/decode engine
# ---------------------------------------------------------------------------


class NumericQuantizer:
    """
    Converts continuous float values to/from integer bin indices.

    Parameters
    ----------
    edges : list of float
        The n+1 boundary values defining n bins.  Must be strictly ascending.
    centres : list of float
        The representative float value for each bin (used for decoding).
        Must have length == len(edges) - 1.
    prefix : str
        Token string prefix (e.g. "NUM" → "NUM_000", "ANG" → "ANG_000").
    """

    def __init__(
        self,
        edges: list[float],
        centres: list[float],
        prefix: str = "NUM",
    ) -> None:
        if len(edges) != len(centres) + 1:
            raise ValueError(
                f"edges length ({len(edges)}) must equal centres length ({len(centres)}) + 1."
            )
        self._edges = edges
        self._centres = centres
        self._prefix = prefix
        self._n = len(centres)
        self._pad_width = len(str(self._n - 1))  # for zero-padded token names

    @property
    def num_bins(self) -> int:
        return self._n

    @property
    def prefix(self) -> str:
        return self._prefix

    def token_for_bin(self, bin_idx: int) -> str:
        """Return canonical token string for a bin index."""
        return f"{self._prefix}_{bin_idx:0{self._pad_width}d}"

    def all_token_strings(self) -> list[str]:
        return [self.token_for_bin(i) for i in range(self._n)]

    def encode(self, value: float) -> tuple[int, str]:
        """
        Map a continuous float to (bin_index, token_string).

        Values outside [edges[0], edges[-1]] are clamped to the nearest bin.
        """
        # Clamp to valid range
        value = max(self._edges[0], min(self._edges[-1], value))
        # Binary search for the bin
        idx = bisect.bisect_right(self._edges, value, lo=1, hi=len(self._edges) - 1) - 1
        idx = max(0, min(self._n - 1, idx))
        return idx, self.token_for_bin(idx)

    def decode(self, token_str: str) -> float | None:
        """
        Recover the representative float value from a token string.
        Returns None if the token does not belong to this quantizer.
        """
        if not token_str.startswith(self._prefix + "_"):
            return None
        try:
            idx = int(token_str[len(self._prefix) + 1 :])
        except ValueError:
            return None
        if idx < 0 or idx >= self._n:
            return None
        return self._centres[idx]

    def decode_bin(self, bin_idx: int) -> float:
        return self._centres[bin_idx]


# ---------------------------------------------------------------------------
# Pre-built standard quantizers
# ---------------------------------------------------------------------------


def make_length_quantizer(
    n_bins: int = 256,
    lo: float = 0.0,
    hi: float = 1_000.0,
    log_scale: bool = False,
) -> NumericQuantizer:
    """
    Length / distance quantizer (millimetres).
    Range: [0, 1000 mm] with n_bins equal-width bins by default.
    """
    if log_scale:
        edges, centres = _log_uniform_bins(max(lo, 0.001), hi, n_bins)
    else:
        edges, centres = _uniform_bins(lo, hi, n_bins)
    return NumericQuantizer(edges, centres, prefix="NUM")


def make_angle_quantizer(n_bins: int = 360) -> NumericQuantizer:
    """
    Angle quantizer (degrees, 0-360, 1 resolution by default).
    """
    edges, centres = _uniform_bins(0.0, 360.0, n_bins)
    return NumericQuantizer(edges, centres, prefix="ANG")


def make_ratio_quantizer(n_bins: int = 128) -> NumericQuantizer:
    """
    Ratio / scale factor quantizer (dimensionless, 0-10).
    Used for draft angles, taper ratios, tolerances, etc.
    """
    edges, centres = _uniform_bins(0.0, 10.0, n_bins)
    return NumericQuantizer(edges, centres, prefix="RAT")


# ---------------------------------------------------------------------------
# NumericTokenizer — populates the CADVocabulary with numeric tokens
# ---------------------------------------------------------------------------


class NumericTokenizer:
    """
    Factory that registers all numeric token strings into a CADVocabulary.

    Token layout within the NUMERIC family range:
        bins 0 .. N_LEN-1   → length tokens   (NUM_000 … NUM_255)
        bins N_LEN .. N_LEN+N_ANG-1 → angle tokens (ANG_000 … ANG_359)
        bins (remainder)    → ratio tokens    (RAT_000 … RAT_127)

    These three quantizers are the *standard* set.  Additional numeric domains
    (force, pressure, temperature, …) can be added by calling
    NumericTokenizer.register_quantizer() and then re-populating.
    """

    # Standard quantizers — class-level singletons (created on first use)
    _length_quantizer: NumericQuantizer | None = None
    _angle_quantizer: NumericQuantizer | None = None
    _ratio_quantizer: NumericQuantizer | None = None

    @classmethod
    def length_quantizer(
        cls,
        n_bins: int = 256,
        lo: float = 0.0,
        hi: float = 1_000.0,
    ) -> NumericQuantizer:
        if cls._length_quantizer is None:
            cls._length_quantizer = make_length_quantizer(n_bins, lo, hi)
        return cls._length_quantizer

    @classmethod
    def angle_quantizer(cls, n_bins: int = 360) -> NumericQuantizer:
        if cls._angle_quantizer is None:
            cls._angle_quantizer = make_angle_quantizer(n_bins)
        return cls._angle_quantizer

    @classmethod
    def ratio_quantizer(cls, n_bins: int = 128) -> NumericQuantizer:
        if cls._ratio_quantizer is None:
            cls._ratio_quantizer = make_ratio_quantizer(n_bins)
        return cls._ratio_quantizer

    @classmethod
    def populate(cls, vocab: CADVocabulary) -> None:
        """Register all numeric token strings into the vocabulary."""
        lq = cls.length_quantizer()
        for tok in lq.all_token_strings():
            vocab.register(tok, TokenFamily.NUMERIC, f"Length bin: {lq.decode(tok):.4f} mm")

        aq = cls.angle_quantizer()
        for tok in aq.all_token_strings():
            vocab.register(tok, TokenFamily.NUMERIC, f"Angle bin: {aq.decode(tok):.2f}°")

        rq = cls.ratio_quantizer()
        for tok in rq.all_token_strings():
            vocab.register(tok, TokenFamily.NUMERIC, f"Ratio bin: {rq.decode(tok):.4f}")

    # ---- High-level encode / decode helpers ----

    @classmethod
    def encode_length(cls, value_mm: float) -> tuple[int, str]:
        """Encode a length in millimetres to (bin_index, token_string)."""
        return cls.length_quantizer().encode(value_mm)

    @classmethod
    def decode_length(cls, token_str: str) -> float | None:
        """Decode a NUM_xxx token back to millimetres."""
        return cls.length_quantizer().decode(token_str)

    @classmethod
    def encode_angle(cls, degrees: float) -> tuple[int, str]:
        """Encode an angle in degrees to (bin_index, token_string)."""
        return cls.angle_quantizer().encode(degrees % 360.0)

    @classmethod
    def decode_angle(cls, token_str: str) -> float | None:
        """Decode an ANG_xxx token back to degrees."""
        return cls.angle_quantizer().decode(token_str)

    @classmethod
    def encode_ratio(cls, value: float) -> tuple[int, str]:
        return cls.ratio_quantizer().encode(value)

    @classmethod
    def decode_ratio(cls, token_str: str) -> float | None:
        return cls.ratio_quantizer().decode(token_str)

    # ---- Legacy compatibility  (mirrors data.py's 20-bin scheme) ----

    @classmethod
    def legacy_encode(cls, value: float, num_bins: int = 20) -> tuple[int, str]:
        """
        Backward-compatible encoder that replicates data.py's 20-bin scheme
        (bins at 0.5, 1.0, …, 10.0).  Used by the legacy shim only.
        """
        bins = [round(0.5 + 0.5 * i, 2) for i in range(num_bins)]
        idx = min(range(len(bins)), key=lambda i: abs(bins[i] - value))
        return idx, f"NUM_{idx}"

    @classmethod
    def legacy_decode(cls, token_str: str, num_bins: int = 20) -> float | None:
        if not token_str.startswith("NUM_"):
            return None
        idx = int(token_str[4:])
        bins = [round(0.5 + 0.5 * i, 2) for i in range(num_bins)]
        if 0 <= idx < len(bins):
            return bins[idx]
        return None
