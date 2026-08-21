"""
tests/tokenizer/test_numeric.py
=================================
Unit tests for cadgenesis.tokenizer.numeric.

Coverage:
    - NumericQuantizer: encode, decode, clamp, token strings
    - make_length_quantizer
    - make_angle_quantizer
    - make_ratio_quantizer
    - NumericTokenizer.populate()
    - encode_length / decode_length round-trip
    - encode_angle / decode_angle round-trip
    - legacy 20-bin encode / decode round-trip
    - Bin boundary conditions (exactly on edge, below min, above max)
"""

from __future__ import annotations

import math

import pytest

from cadgenesis.tokenizer.numeric import (
    NumericQuantizer,
    NumericTokenizer,
    _log_uniform_bins,
    _uniform_bins,
    make_angle_quantizer,
    make_length_quantizer,
    make_ratio_quantizer,
)
from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def length_q() -> NumericQuantizer:
    return make_length_quantizer(n_bins=256)


@pytest.fixture
def angle_q() -> NumericQuantizer:
    return make_angle_quantizer(n_bins=360)


@pytest.fixture
def ratio_q() -> NumericQuantizer:
    return make_ratio_quantizer(n_bins=128)


# ---------------------------------------------------------------------------
# _uniform_bins / _log_uniform_bins
# ---------------------------------------------------------------------------


class TestBinGenerators:
    def test_uniform_bins_count(self):
        edges, centres = _uniform_bins(0, 100, 10)
        assert len(edges) == 11
        assert len(centres) == 10

    def test_uniform_bins_range(self):
        edges, _centres = _uniform_bins(0, 100, 10)
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(100.0)

    def test_uniform_bins_equidistant(self):
        edges, _centres = _uniform_bins(0, 100, 10)
        gaps = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        assert all(math.isclose(g, 10.0) for g in gaps)

    def test_uniform_centres_are_midpoints(self):
        edges, centres = _uniform_bins(0, 100, 4)
        for i, c in enumerate(centres):
            expected = (edges[i] + edges[i + 1]) / 2
            assert c == pytest.approx(expected)

    def test_log_uniform_bins_count(self):
        edges, centres = _log_uniform_bins(0.1, 1000, 10)
        assert len(edges) == 11
        assert len(centres) == 10

    def test_log_uniform_requires_positive_lo(self):
        with pytest.raises(ValueError, match="log-uniform"):
            _log_uniform_bins(0, 100, 10)

    def test_log_uniform_denser_at_low_end(self):
        edges, _centres = _log_uniform_bins(1, 1000, 10)
        gaps = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        # Gaps should be strictly increasing (log-uniform = more bins near lo)
        assert gaps[0] < gaps[-1]


# ---------------------------------------------------------------------------
# NumericQuantizer
# ---------------------------------------------------------------------------


class TestNumericQuantizer:
    def test_num_bins(self, length_q):
        assert length_q.num_bins == 256

    def test_token_for_bin_format(self, length_q):
        # Should be zero-padded to consistent width
        tok = length_q.token_for_bin(0)
        assert tok.startswith("NUM_")
        assert "_" in tok

    def test_all_token_strings_length(self, length_q):
        toks = length_q.all_token_strings()
        assert len(toks) == 256

    def test_all_token_strings_unique(self, length_q):
        toks = length_q.all_token_strings()
        assert len(toks) == len(set(toks))

    def test_encode_returns_valid_bin(self, length_q):
        idx, tok = length_q.encode(50.0)
        assert 0 <= idx < length_q.num_bins
        assert tok.startswith("NUM_")

    def test_encode_decode_round_trip(self, length_q):
        original = 123.456
        _, tok = length_q.encode(original)
        decoded = length_q.decode(tok)
        # Should be within one bin-width of the original
        bin_width = 1000.0 / 256
        assert abs(decoded - original) <= bin_width

    def test_encode_clamps_below_min(self, length_q):
        idx_below, _ = length_q.encode(-999.0)
        idx_min, _ = length_q.encode(0.0)
        assert idx_below == idx_min  # clamped to first bin

    def test_encode_clamps_above_max(self, length_q):
        idx_above, _ = length_q.encode(9999.0)
        idx_max, _ = length_q.encode(1000.0)
        assert idx_above == idx_max  # clamped to last bin

    def test_decode_wrong_prefix_returns_none(self, length_q):
        assert length_q.decode("ANG_001") is None
        assert length_q.decode("GARBAGE") is None

    def test_decode_out_of_range_returns_none(self, length_q):
        # Manually craft a token with idx > num_bins
        bad_tok = f"NUM_{length_q.num_bins + 100}"
        assert length_q.decode(bad_tok) is None

    def test_encode_monotone(self, length_q):
        """Larger values should encode to equal or higher bin indices."""
        values = [0, 100, 250, 500, 750, 1000]
        idxs = [length_q.encode(v)[0] for v in values]
        assert idxs == sorted(idxs)


# ---------------------------------------------------------------------------
# Angle quantizer
# ---------------------------------------------------------------------------


class TestAngleQuantizer:
    def test_prefix_is_ang(self, angle_q):
        assert angle_q.prefix == "ANG"

    def test_360_bins(self, angle_q):
        assert angle_q.num_bins == 360

    def test_encode_360_wraps(self, angle_q):
        # encode_angle wraps modulo 360 before calling quantizer
        idx_0, _ = angle_q.encode(0.0)
        idx_360, _ = NumericTokenizer.encode_angle(360.0)
        # 360° mod 360 = 0° — should map to same bin
        assert idx_0 == idx_360

    def test_decode_angle(self, angle_q):
        _, tok = angle_q.encode(90.0)
        decoded = angle_q.decode(tok)
        assert abs(decoded - 90.0) <= 1.0  # within 1°


# ---------------------------------------------------------------------------
# NumericTokenizer.populate
# ---------------------------------------------------------------------------


class TestNumericTokenizerPopulate:
    def test_populates_vocabulary(self):
        v = CADVocabulary()
        NumericTokenizer.populate(v)
        numeric_toks = v.tokens_in_family(TokenFamily.NUMERIC)
        assert len(numeric_toks) > 0

    def test_num_prefix_tokens_present(self):
        v = CADVocabulary()
        NumericTokenizer.populate(v)
        # All NUM_xxx tokens should be registered
        lq = NumericTokenizer.length_quantizer()
        for tok in lq.all_token_strings():
            assert tok in v

    def test_ang_prefix_tokens_present(self):
        v = CADVocabulary()
        NumericTokenizer.populate(v)
        aq = NumericTokenizer.angle_quantizer()
        for tok in aq.all_token_strings():
            assert tok in v

    def test_all_numeric_are_in_numeric_family(self):
        v = CADVocabulary()
        NumericTokenizer.populate(v)
        for tok in NumericTokenizer.length_quantizer().all_token_strings():
            assert v.family_of(tok) == TokenFamily.NUMERIC


# ---------------------------------------------------------------------------
# encode/decode helpers (module-level)
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_encode_decode_length(self):
        for mm in [0.0, 1.0, 50.5, 250.0, 999.9]:
            _, tok = NumericTokenizer.encode_length(mm)
            decoded = NumericTokenizer.decode_length(tok)
            bin_width = 1000.0 / 256
            assert decoded is not None
            assert abs(decoded - mm) <= bin_width

    def test_encode_decode_angle(self):
        for deg in [0.0, 45.0, 90.0, 180.0, 270.0, 359.0]:
            _, tok = NumericTokenizer.encode_angle(deg)
            decoded = NumericTokenizer.decode_angle(tok)
            assert decoded is not None
            assert abs(decoded - deg) <= 1.0  # within 1°

    def test_encode_ratio(self):
        _, tok = NumericTokenizer.encode_ratio(1.5)
        decoded = NumericTokenizer.decode_ratio(tok)
        assert decoded is not None
        assert abs(decoded - 1.5) <= 10.0 / 128


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------


class TestLegacyCompat:
    def test_legacy_encode_20_bins(self):
        for val in [0.5, 1.0, 5.0, 10.0]:
            idx, tok = NumericTokenizer.legacy_encode(val, num_bins=20)
            assert 0 <= idx < 20
            assert tok == f"NUM_{idx}"

    def test_legacy_decode(self):
        for i in range(20):
            tok = f"NUM_{i}"
            decoded = NumericTokenizer.legacy_decode(tok, num_bins=20)
            assert decoded is not None
            expected = round(0.5 + 0.5 * i, 2)
            assert decoded == pytest.approx(expected)

    def test_legacy_decode_invalid_prefix(self):
        assert NumericTokenizer.legacy_decode("ANG_001") is None

    def test_legacy_decode_out_of_range(self):
        assert NumericTokenizer.legacy_decode("NUM_999", num_bins=20) is None
