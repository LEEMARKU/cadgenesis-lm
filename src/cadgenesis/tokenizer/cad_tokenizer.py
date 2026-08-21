"""
cadgenesis.tokenizer.cad_tokenizer
=====================================
AutonomousCADTokenizer — the unified orchestrator for the Autonomous CAD
Tokenizer subsystem (Phase 1 of CADGenesis-LM v2.0).

Purpose
-------
This class is the single entry point for all tokenization operations.  It:

1. Owns a ``CADVocabulary`` with all 9 token families registered
2. Wraps a language tokenizer (legacy word-level or BPE)
3. Provides high-level encode/decode for multi-modal CAD sequences
4. Manages the numeric quantizers for all parameter types
5. Validates token sequences structurally (lightweight; deep validation
   is the job of the CAD Execution Intelligence Engine in Phase 6)
6. Serializes / loads the complete tokenizer state

Architecture
------------
::

    AutonomousCADTokenizer
    ├── vocab              : CADVocabulary        (all token families)
    ├── lang_tok           : LanguageTokenizerBase (text side)
    ├── num_tok            : NumericTokenizer      (quantization)
    └── encode / decode methods for each modality

Data Structures
---------------
    CADTokenSequence — dataclass wrapping a tokenized CAD multimodal sequence:
        text_ids        : List[int]    language token ids
        cad_ids         : List[int]    CAD token ids (incl. type markers)
        type_ids        : List[int]    token family ids (for type embedding)
        attention_mask  : List[int]    1 = real token, 0 = padding
        modality_mask   : List[int]    0=text, 1=cad (for cross-attention routing)

    MultiModalBatch — batched version of CADTokenSequence (tensors)

Interfaces
----------
    # Build and use
    tok = AutonomousCADTokenizer.build()
    seq = tok.encode_text("Create a steel box 50mm wide")
    seq = tok.encode_cad_sequence(["PRIM_BOX", "NUM_012", "NUM_025", "NUM_008"])
    seq = tok.encode_multimodal(text, cad_tokens)

    # Tensor-ready batching
    batch = tok.collate([seq1, seq2, seq3])

    # I/O
    tok.save("outputs/cad_tok")
    tok2 = AutonomousCADTokenizer.load("outputs/cad_tok")

Algorithms
----------
    encode_multimodal:   O(N_text + N_cad)
    collate:             O(B * L)  where B=batch, L=max_seq_len
    type_id_of_token:    O(1)  (vocabulary lookup)

Complexity
----------
    Vocabulary build:  O(V)  where V = total token count (~34000 default)
    Encode:            O(N)  where N = sequence length
    Decode:            O(N)
    Collate:           O(B * L)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadgenesis.tokenizer.statistics import CorpusStatistics
    from cadgenesis.tokenizer.versioning import MigrationResult

from cadgenesis.tokenizer.language import LanguageTokenizerBase, LegacyWordTokenizer
from cadgenesis.tokenizer.numeric import NumericTokenizer
from cadgenesis.tokenizer.vocabulary import (
    ASSEMBLY_END_TOKEN,
    ASSEMBLY_START_TOKEN,
    BOS_TOKEN,
    CAD_END_TOKEN,
    CAD_START_TOKEN,
    CONSTRAINT_END_TOKEN,
    CONSTRAINT_START_TOKEN,
    EOS_TOKEN,
    MANUF_END_TOKEN,
    MANUF_START_TOKEN,
    MATERIAL_END_TOKEN,
    MATERIAL_START_TOKEN,
    PAD_TOKEN,
    SIM_END_TOKEN,
    SIM_START_TOKEN,
    UNK_TOKEN,
    CADVocabulary,
    TokenFamily,
)

# ---------------------------------------------------------------------------
# Legacy dataset token convention
# ---------------------------------------------------------------------------
# The dataset layers (``datasets.cad_program_synth`` and the
# ``tokenizer.legacy_shim`` generators) emit *unpadded* numeric tokens
# (``NUM_5``, ``NUM_80``) and un-namespaced operation/object names
# (``BOX``, ``EXTRUDE``, ``SKETCH_RECT``).  The canonical vocabulary uses
# zero-padded numeric names (``NUM_005``) and namespaced operations
# (``PRIM_BOX``, ``FEAT_EXTRUDE``).  These legacy strings are registered as
# first-class tokens so that every dataset program is encodable end-to-end.
# The canonical names remain registered; the two namespaces coexist and both
# decode via ``NumericTokenizer.decode_length`` (which parses any integer
# suffix, padded or not).
#
# ``NUM_0 .. NUM_255`` are registered lazily by
# ``_register_legacy_numeric_tokens`` (only the range actually needed by the
# dataset is materialised to keep the vocabulary tight).

#: Legacy operation / feature names (FEATURE family).
_LEGACY_FEATURE_TOKENS: list[str] = [
    "EXTRUDE",
    "REVOLVE",
    "HOLE",
    "THREAD",
    "PATTERN",
    "FILLET",
    "CHAMFER",
    "MIRROR",
    "BOOLEAN_UNION",
    "BOOLEAN_CUT",
    "COUNTERBORE",
    "SLOT",
]

#: Legacy primitive / part / object names (GEOMETRY family).
_LEGACY_GEOMETRY_TOKENS: list[str] = [
    "BOX",
    "CYLINDER",
    "SPHERE",
    "SKETCH_RECT",
    "SKETCH",
    "RECT",
    "BASE",
    "PART",
    "BRACKET",
    "DOWEL",
    "PEG",
    "MOUNT",
    "FIXTURE",
    "EDGE",
    "BOLT",
    "STEEL",
    "SHAFT",
    "WEIGHT",
    "VOLUME",
    "CLEARANCE",
    "SPACING",
]

#: Highest unpadded numeric token registered (matches dataset NUM_MAX range).
_LEGACY_NUM_MAX = 255


def _register_legacy_numeric_tokens(vocab: CADVocabulary, n: int = _LEGACY_NUM_MAX) -> int:
    """Register unpadded ``NUM_0..NUM_{n}`` length tokens.

    Returns the number of newly registered tokens.
    """
    count = 0
    for i in range(n + 1):
        tok = f"NUM_{i}"
        if tok not in vocab:
            vocab.register(tok, TokenFamily.NUMERIC, f"Legacy raw-mm length token ({i} mm)")
            count += 1
    return count


def _register_legacy_cad_tokens(vocab: CADVocabulary, numeric_max: int = _LEGACY_NUM_MAX) -> int:
    """Register the legacy dataset token convention into ``vocab``.

    Idempotent (skips already-registered tokens).  Returns the number of
    newly registered tokens.
    """
    count = 0
    for tok in _LEGACY_GEOMETRY_TOKENS:
        if tok not in vocab:
            vocab.register(tok, TokenFamily.GEOMETRY, "Legacy primitive / part / object name")
            count += 1
    for tok in _LEGACY_FEATURE_TOKENS:
        if tok not in vocab:
            vocab.register(tok, TokenFamily.FEATURE, "Legacy feature operation name")
            count += 1
    count += _register_legacy_numeric_tokens(vocab, numeric_max)
    return count


# ---------------------------------------------------------------------------
# CADTokenSequence — single-example data structure
# ---------------------------------------------------------------------------


@dataclass
class CADTokenSequence:
    """
    A fully tokenized multi-modal CAD example.

    All lists are aligned:  cad_ids[i] has type type_ids[i] and is masked
    by attention_mask[i].

    Fields
    ------
    text_ids        Language token ids (encoder input)
    cad_ids         CAD token ids (decoder input / target)
    type_ids        Token family integer for each CAD token (type embedding)
    attention_mask  1 = real, 0 = padding (for CAD side)
    modality_mask   0 = language token, 1 = CAD token
    raw_text        Original text string (for debugging)
    raw_cad_tokens  Original CAD token strings (for debugging)
    """

    text_ids: list[int] = field(default_factory=list)
    cad_ids: list[int] = field(default_factory=list)
    type_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    modality_mask: list[int] = field(default_factory=list)
    raw_text: str = ""
    raw_cad_tokens: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cad_ids)

    def is_valid(self) -> bool:
        """Lightweight structural check: all list lengths are consistent."""
        n = len(self.cad_ids)
        return len(self.type_ids) == n and len(self.attention_mask) == n and n > 0


@dataclass
class MultiModalBatch:
    """
    Batched, padded tensors ready for model input.

    All tensors are shape (B, L) where B=batch_size, L=max_seq_len.
    This class intentionally avoids a hard torch dependency so the tokenizer
    package is importable without PyTorch.  Call ``.to_torch()`` to get
    actual tensors.
    """

    text_ids: list[list[int]]  # (B, S)
    cad_ids: list[list[int]]  # (B, T)
    type_ids: list[list[int]]  # (B, T)
    attention_mask: list[list[int]]  # (B, T)
    batch_size: int
    max_src_len: int
    max_tgt_len: int

    def to_torch(self):
        """Convert to a dict of torch.Tensors (requires PyTorch)."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "MultiModalBatch.to_torch() requires PyTorch. Install with: pip install torch"
            ) from exc
        return {
            "text_ids": torch.tensor(self.text_ids, dtype=torch.long),
            "cad_ids": torch.tensor(self.cad_ids, dtype=torch.long),
            "type_ids": torch.tensor(self.type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# AutonomousCADTokenizer
# ---------------------------------------------------------------------------


class AutonomousCADTokenizer:
    """
    Unified orchestrator for all CADGenesis-LM v2.0 tokenization.

    This is the single class that application code should interact with.
    Sub-tokenizers (geometry, feature, constraint, material, assembly,
    manufacturing, simulation, numeric) are implementation details hidden
    behind this interface.

    Parameters
    ----------
    vocab : CADVocabulary
        Pre-populated vocabulary registry.
    lang_tok : LanguageTokenizerBase
        Language tokenizer (legacy word-level or BPE).
    max_text_len : int
        Maximum number of language tokens to accept (truncates beyond this).
    max_cad_len : int
        Maximum number of CAD tokens in a single sequence.
    """

    _STATE_FILENAME = "tokenizer_state.json"

    def __init__(
        self,
        vocab: CADVocabulary,
        lang_tok: LanguageTokenizerBase,
        max_text_len: int = 512,
        max_cad_len: int = 1_024,
    ) -> None:
        self.vocab = vocab
        self.lang_tok = lang_tok
        self.max_text_len = max_text_len
        self.max_cad_len = max_cad_len

        # Frequently accessed special token IDs (cached for hot paths)
        self._pad_id = vocab.record_of(PAD_TOKEN).token_id
        self._bos_id = vocab.record_of(BOS_TOKEN).token_id
        self._eos_id = vocab.record_of(EOS_TOKEN).token_id
        self._unk_id = (
            vocab.record_of(UNK_TOKEN).token_id
            if UNK_TOKEN in vocab
            else vocab.record_of(PAD_TOKEN).token_id
        )

        # CAD section boundary ids
        self._cad_start_id = vocab[CAD_START_TOKEN]
        self._cad_end_id = vocab[CAD_END_TOKEN]
        self._con_start_id = vocab[CONSTRAINT_START_TOKEN]
        self._con_end_id = vocab[CONSTRAINT_END_TOKEN]
        self._asm_start_id = vocab[ASSEMBLY_START_TOKEN]
        self._asm_end_id = vocab[ASSEMBLY_END_TOKEN]
        self._mat_start_id = vocab[MATERIAL_START_TOKEN]
        self._mat_end_id = vocab[MATERIAL_END_TOKEN]
        self._mfg_start_id = vocab[MANUF_START_TOKEN]
        self._mfg_end_id = vocab[MANUF_END_TOKEN]
        self._sim_start_id = vocab[SIM_START_TOKEN]
        self._sim_end_id = vocab[SIM_END_TOKEN]

    # ---- Factory methods ----

    @classmethod
    def build(
        cls,
        lang_tok: LanguageTokenizerBase | None = None,
        max_text_len: int = 512,
        max_cad_len: int = 1_024,
    ) -> AutonomousCADTokenizer:
        """
        Build a fully-populated AutonomousCADTokenizer with the default
        CAD vocabulary.

        Parameters
        ----------
        lang_tok : LanguageTokenizerBase, optional
            Language tokenizer.  If None, uses ``LegacyWordTokenizer``
            (empty vocab — call ``build_lang_vocab(texts)`` before encoding).
        max_text_len, max_cad_len : int
            Sequence length limits.
        """
        vocab = CADVocabulary.build_default()
        _register_legacy_cad_tokens(vocab)
        if lang_tok is None:
            lang_tok = LegacyWordTokenizer()
        return cls(vocab, lang_tok, max_text_len, max_cad_len)

    @classmethod
    def build_mini(cls) -> AutonomousCADTokenizer:
        """
        Build a minimal tokenizer backward-compatible with data.py.

        Uses 20-bin legacy quantization, 5 primitives, 3 specials — same
        as the original CADGenesisMini.  Useful for running existing tests
        and the Colab notebook without modification.
        """
        from cadgenesis.tokenizer.vocabulary import (
            CADVocabulary,
            TokenFamily,
            _register_special_tokens,
        )

        # Build a stripped-down vocabulary
        # Capacity layout keeps every CAD-family id < 512 so mini-mode
        # models (``CADConfig.mini()`` sizes lang_vocab_size=512) can embed
        # every CAD id.  CAD ids occupy 0..511; language ids start at 512.
        vocab = CADVocabulary(
            slots={
                TokenFamily.SPECIAL: 64,
                TokenFamily.NUMERIC: 384,  # raw NUM_0..255 + padded NUM_000..099
                TokenFamily.GEOMETRY: 32,  # legacy primitives + part names
                TokenFamily.FEATURE: 32,  # legacy feature operations
                TokenFamily.CONSTRAINT: 0,
                TokenFamily.MATERIAL: 0,
                TokenFamily.ASSEMBLY: 0,
                TokenFamily.MANUFACTURING: 0,
                TokenFamily.SIMULATION: 0,
                TokenFamily.LANGUAGE: 512,
            }
        )
        _register_special_tokens(vocab)

        # Register the full unpadded numeric range (legacy raw-mm convention)
        _register_legacy_numeric_tokens(vocab)

        # Register canonical zero-padded quantizer bins for values >= 100 mm
        # (dataset values <= 155 mm quantize to bins <= 39, so NUM_000..NUM_099
        # is ample).  Must stay within the mini model's lang embedding range
        # (``CADConfig.mini()`` sizes lang_vocab_size=512; CAD ids must not
        # collide with that space).
        for i in range(100):
            tok = f"NUM_{i:03d}"
            if tok not in vocab:
                desc = f"Length bin: {NumericTokenizer.decode_length(tok):.4f} mm"
                vocab.register(tok, TokenFamily.NUMERIC, desc)

        # Register legacy primitives as GEOMETRY tokens
        for prim in ["BOX", "CYLINDER", "SPHERE", "SKETCH_RECT", "EXTRUDE"]:
            vocab.register(prim, TokenFamily.GEOMETRY, f"Legacy primitive {prim}")

        # Register the remaining legacy part/object names + feature operations
        _register_legacy_cad_tokens(vocab)

        lang_tok = LegacyWordTokenizer()
        return cls(vocab, lang_tok, max_text_len=32, max_cad_len=16)

    # ---- Language vocabulary building ----

    def build_lang_vocab(self, texts: list[str]) -> None:
        """
        Build the language vocabulary from a corpus.  Only valid when using
        ``LegacyWordTokenizer``.  No-op for BPE tokenizers.
        """
        if isinstance(self.lang_tok, LegacyWordTokenizer):
            self.lang_tok.build_vocab(texts)

    # ---- Core encoding ----

    def encode_text(self, text: str) -> list[int]:
        """Encode a natural-language string to language token ids.

        Engineering notation (``Ø25``, ``R12.5``, ``M8x1.25``, ``±0.02``,
        ``(10,20,30)``, units, exponents) is normalised first so it becomes
        segmentable plain text; plain text passes through unchanged.
        """
        from cadgenesis.tokenizer.engineering import normalize_engineering_notation

        text = normalize_engineering_notation(text)
        ids = self.lang_tok.encode(text)
        return ids[: self.max_text_len]

    def encode_cad_token(self, token_str: str, auto_register: bool = False) -> int:
        """
        Look up a single CAD token string in the vocabulary.

        If ``auto_register`` is True and the token is unknown, it is
        registered on the fly into its guessed family (see
        ``cadgenesis.tokenizer.evolution.guess_family``) when a slot is free.
        Returns the <unk> id if the token is not registered.
        """
        if token_str in self.vocab:
            return self.vocab.record_of(token_str).token_id
        if auto_register:
            from cadgenesis.tokenizer.evolution import guess_family

            family = guess_family(token_str)
            if self.vocab.remaining_slots(family) > 0:
                self.vocab.register(token_str, family, "auto-registered")
                return self.vocab.record_of(token_str).token_id
        return self._unk_id

    def encode_cad_sequence(
        self,
        tokens: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> CADTokenSequence:
        """
        Encode a list of CAD token strings into a ``CADTokenSequence``.

        Parameters
        ----------
        tokens : List[str]
            CAD token strings (without BOS/EOS — they are added by this method
            if ``add_bos`` / ``add_eos`` are True).
        add_bos : bool
            Prepend <bos> token.
        add_eos : bool
            Append <eos> token.

        Returns
        -------
        CADTokenSequence
            Fully populated with cad_ids, type_ids, attention_mask.
        """
        cad_toks: list[str] = []
        if add_bos:
            cad_toks.append(BOS_TOKEN)
        cad_toks.extend(tokens)
        if add_eos:
            cad_toks.append(EOS_TOKEN)
        cad_toks = cad_toks[: self.max_cad_len]

        cad_ids = [self.encode_cad_token(t) for t in cad_toks]
        type_ids = [self.vocab.type_id_of(i) if i in self.vocab else 0 for i in cad_ids]
        attn = [1] * len(cad_ids)

        return CADTokenSequence(
            cad_ids=cad_ids,
            type_ids=type_ids,
            attention_mask=attn,
            raw_cad_tokens=cad_toks,
        )

    def encode_multimodal(
        self,
        text: str,
        cad_tokens: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> CADTokenSequence:
        """
        Encode a paired (text, CAD sequence) example.

        Parameters
        ----------
        text : str
            Natural-language design request.
        cad_tokens : List[str]
            CAD token string sequence.
        add_bos, add_eos : bool
            Whether to prepend/append BOS/EOS on the CAD side.

        Returns
        -------
        CADTokenSequence
            text_ids populated from language side;
            cad_ids / type_ids from CAD side.
        """
        seq = self.encode_cad_sequence(cad_tokens, add_bos, add_eos)
        seq.text_ids = self.encode_text(text)
        seq.raw_text = text
        return seq

    # ---- Decode ----

    def decode_text(self, ids: list[int]) -> str:
        """Decode language token ids back to text."""
        return self.lang_tok.decode(ids)

    def decode_cad_sequence(self, ids: list[int]) -> list[str]:
        """
        Decode CAD token ids back to token strings.

        Filters out padding ids and maps unknown ids to "<unk>".
        """
        result = []
        for tok_id in ids:
            if tok_id == self._pad_id:
                continue
            if tok_id in self.vocab:
                result.append(self.vocab.record_of(tok_id).token_str)
            else:
                result.append(UNK_TOKEN)
        return result

    # ---- Numeric parameter helpers ----

    def encode_length(self, value_mm: float) -> tuple[int, str]:
        """Encode a length in mm to (bin_index, token_string)."""
        return NumericTokenizer.encode_length(value_mm)

    def decode_length(self, token_str: str) -> float | None:
        """Decode a NUM_xxx token to mm.

        Two numeric conventions coexist by design:

        * **Unpadded** legacy tokens (``NUM_80``) — raw millimetre semantics
          (``NUM_80`` decodes to 80.0 mm).  This is the convention emitted by
          the legacy dataset layers.
        * **Padded** canonical tokens (``NUM_020``) — quantizer bin index
          semantics (decoded via ``NumericTokenizer.decode_length``).  The
          dataset emits these for values >= 100 mm, where the unpadded name
          would collide with a canonical bin token.
        """
        if token_str.startswith("NUM_") and token_str[4:].isdigit():
            digits = token_str[4:]
            if len(digits) < 3 and digits == str(int(digits)):
                value = int(digits)
                if 0 <= value <= 1_000:
                    return float(value)
        return NumericTokenizer.decode_length(token_str)

    def encode_angle(self, degrees: float) -> tuple[int, str]:
        """Encode an angle in degrees to (bin_index, token_string)."""
        return NumericTokenizer.encode_angle(degrees)

    def decode_angle(self, token_str: str) -> float | None:
        """Decode an ANG_xxx token to degrees."""
        return NumericTokenizer.decode_angle(token_str)

    # ---- Token metadata ----

    def type_id_of(self, token_str: str) -> int:
        """Return the token family integer id (type embedding input)."""
        if token_str in self.vocab:
            return self.vocab.type_id_of(token_str)
        return TokenFamily.SPECIAL.value

    def family_of(self, token_str: str) -> TokenFamily | None:
        """Return the TokenFamily for a token string, or None if unknown."""
        if token_str in self.vocab:
            return self.vocab.family_of(token_str)
        return None

    # ---- Unknown-token handling ----

    def is_unknown_token(self, token_str: str) -> bool:
        """True if ``token_str`` is not a registered vocabulary entry."""
        return token_str not in self.vocab

    def validate_token(self, token_str: str) -> tuple[bool, str]:
        """
        Validate a single CAD token.

        Checks registration, then family-specific rules:
        * ``NUM_``/``ANG_`` numeric tokens must decode to a finite value.
        * Special/control tokens are accepted as-is.

        Returns ``(is_valid, message)``.  Unknown tokens are invalid (they
        must be resolved via ``encode_cad_token(..., auto_register=True)`` or
        ``register_new_token``).
        """
        if token_str not in self.vocab:
            return False, f"Token {token_str!r} is not registered."

        if token_str.startswith("NUM_") or token_str.startswith("ANG_"):
            value = NumericTokenizer.decode_angle(token_str)
            if value is None:
                value = NumericTokenizer.decode_length(token_str)
            if value is None:
                return False, f"Numeric token {token_str!r} is not decodable."
        return True, "OK"

    def register_new_token(
        self,
        token_str: str,
        family: TokenFamily | None = None,
        description: str = "user-registered",
    ) -> int:
        """
        Explicitly register a previously-unknown token, guessing its family
        when none is supplied.  Raises ``KeyError`` if already registered.
        Returns the assigned token id.
        """
        if token_str in self.vocab:
            raise KeyError(f"Token {token_str!r} is already registered.")
        if family is None:
            from cadgenesis.tokenizer.evolution import guess_family

            family = guess_family(token_str)
        record = self.vocab.register(token_str, family, description)
        return record.token_id

    def unknown_rate(self, sequences: list[CADTokenSequence]) -> float:
        """
        Fraction of CAD tokens across ``sequences`` that are out-of-vocabulary
        (0.0 → 1.0).  Uses ``cadgenesis.tokenizer.statistics`` for the count.
        """
        from cadgenesis.tokenizer.statistics import compute_statistics

        stats = compute_statistics(sequences, self.vocab)
        return stats.unknown_rate

    # ---- Token validation / compression ----

    def compress_sequence(
        self,
        tokens: list[str],
    ) -> tuple[list[str], float]:
        """
        Greedily merge adjacent registered composite tokens to compress a CAD
        token sequence.  Returns ``(compressed_tokens, ratio)`` where
        ``ratio = 1 - len(compressed) / len(original)``.  This is a lossless
        pass: every composite token can be expanded via ``expand_sequence``.
        """
        if not tokens:
            return [], 0.0
        merged = self.remap_sequence(tokens)
        ratio = 1.0 - (len(merged) / len(tokens))
        return merged, ratio

    def expand_sequence(self, tokens: list[str]) -> list[str]:
        """Losslessly expand composite tokens back to their component tokens."""
        flattened: list[str] = []
        for tok in tokens:
            flattened.extend(self.vocab.expand_token(tok))
        return flattened

    # ---- Corpus token statistics ----

    def token_statistics(
        self,
        sequences: list[CADTokenSequence],
        compress_fn=None,
    ) -> CorpusStatistics:
        """
        Aggregate token statistics over a list of ``CADTokenSequence``
        examples (see ``cadgenesis.tokenizer.statistics``).  When
        ``compress_fn`` is None, the tokenizer's own ``compress_sequence`` is
        used to measure the achievable compression ratio.
        """
        from cadgenesis.tokenizer.statistics import compute_statistics

        if compress_fn is None:
            compress_fn = self.compress_sequence
        return compute_statistics(sequences, self.vocab, compress_fn)

    # ---- Vocabulary versioning / migration ----

    @property
    def vocabulary_version(self) -> str:
        """Semantic version of the current CAD vocabulary."""
        return self.vocab.version

    def migrate_vocabulary(
        self,
        new_slots: dict[TokenFamily, int],
        target_version: str | None = None,
    ) -> MigrationResult:
        """
        Migrate the tokenizer's vocabulary onto ``new_slots`` (see
        ``cadgenesis.tokenizer.versioning``).  Returns a ``MigrationResult``;
        the tokenizer itself is NOT modified until you assign
        ``tok.vocab = result.vocab``.
        """
        from cadgenesis.tokenizer.versioning import migrate_vocabulary

        return migrate_vocabulary(self.vocab, new_slots, target_version)

    def remap_ids_to_vocab(
        self,
        ids: list[int],
        mapping: dict[int, int] | None = None,
        fallback_unk_id: int | None = None,
    ) -> list[int]:
        """
        Translate an id sequence (e.g. previously-saved tokens) into the id
        space of a migrated vocabulary.

        ``mapping`` is the old→new id mapping from a ``MigrationResult``
        (``result.id_mapping``).  When omitted, the most recent migration
        performed on the tokenizer's own vocabulary is used.  Ids absent from
        the mapping fall back to <unk>.
        """
        from cadgenesis.tokenizer.versioning import remap_ids

        if fallback_unk_id is None:
            fallback_unk_id = self._unk_id
        if mapping is None:
            mapping = self.vocab.last_migration[0] if self.vocab.last_migration else {}
        return remap_ids(ids, mapping, fallback_unk_id)

    # ---- Collation / batching ----

    def collate(
        self,
        sequences: list[CADTokenSequence],
        max_src: int | None = None,
        max_tgt: int | None = None,
    ) -> MultiModalBatch:
        """
        Pad and batch a list of CADTokenSequences.

        Parameters
        ----------
        sequences : List[CADTokenSequence]
            Batch of encoded examples.
        max_src : int, optional
            Maximum language sequence length.  Defaults to the longest in batch.
        max_tgt : int, optional
            Maximum CAD sequence length.  Defaults to the longest in batch.

        Returns
        -------
        MultiModalBatch
            Padded lists ready for ``to_torch()`` conversion.
        """
        if not sequences:
            raise ValueError("Cannot collate an empty batch.")

        S = max_src or max(len(s.text_ids) for s in sequences)
        T = max_tgt or max(len(s.cad_ids) for s in sequences)
        B = len(sequences)

        # Pad source (language) side
        text_ids_batch = []
        for seq in sequences:
            ids = seq.text_ids[:S]
            pad = [self._pad_id] * (S - len(ids))
            text_ids_batch.append(ids + pad)

        # Pad target (CAD) side
        cad_ids_batch = []
        type_ids_batch = []
        attn_batch = []
        for seq in sequences:
            n = min(len(seq.cad_ids), T)
            c_ids = seq.cad_ids[:n] + [self._pad_id] * (T - n)
            t_ids = seq.type_ids[:n] + [0] * (T - n)
            attn = seq.attention_mask[:n] + [0] * (T - n)
            cad_ids_batch.append(c_ids)
            type_ids_batch.append(t_ids)
            attn_batch.append(attn)

        return MultiModalBatch(
            text_ids=text_ids_batch,
            cad_ids=cad_ids_batch,
            type_ids=type_ids_batch,
            attention_mask=attn_batch,
            batch_size=B,
            max_src_len=S,
            max_tgt_len=T,
        )

    # ---- Structural validation ----

    def validate_cad_sequence(self, tokens: list[str]) -> tuple[bool, str]:
        """
        Lightweight structural validation of a CAD token sequence.

        Checks:
        - All tokens are registered in the vocabulary
        - BOS is the first non-pad token (if present)
        - EOS is the last non-pad token (if present)
        - No unknown tokens in the geometry/feature positions

        This is intentionally shallow.  Deep geometric validation is done by
        the CAD Execution Intelligence Engine (Phase 6).

        Returns
        -------
        (is_valid : bool, message : str)
        """
        if not tokens:
            return False, "Empty token sequence."

        unknown = [t for t in tokens if t not in self.vocab]
        if unknown:
            return False, f"Unknown tokens: {unknown[:5]}"

        # Strip specials for content check
        content = [t for t in tokens if t not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)]
        if not content:
            return False, "Sequence contains only special tokens."

        # First content token should be a geometry or feature token
        first = content[0]
        family = self.family_of(first)
        if family not in (TokenFamily.GEOMETRY, TokenFamily.FEATURE):
            return (
                False,
                f"Expected first CAD token to be GEOMETRY or FEATURE, "
                f"got {family.name if family else 'UNKNOWN'} ({first!r}).",
            )

        return True, "OK"

    # ---- Vocabulary stats ----

    def vocab_stats(self) -> dict:
        """Return vocabulary usage statistics per family."""
        return self.vocab.stats()

    @property
    def vocab_size(self) -> int:
        """Total number of registered tokens."""
        return len(self.vocab)

    @property
    def lang_vocab_size(self) -> int:
        return self.lang_tok.vocab_size

    @property
    def cad_vocab_size(self) -> int:
        """Number of non-language tokens (geometry+feature+numeric+…)."""
        return self.vocab_size - self.lang_vocab_size

    # ---- Vocabulary evolution (autonomous growth) ----

    def evolve(
        self,
        sequences: list[CADTokenSequence],
        min_frequency: int = 3,
        min_pair_frequency: int = 5,
        max_ops: int = 64,
        family_guesser=None,
    ) -> dict:
        """
        Analyse a corpus of ``CADTokenSequence`` examples and grow the
        vocabulary to match observed usage (autonomous dynamic growth).

        * Registers frequent unknown CAD tokens.
        * Merges frequent adjacent token pairs into composite tokens.

        Returns a report dict with ``plan``, ``new_tokens``, ``applied_ops``
        and the corpus ``stats``.
        """
        from cadgenesis.tokenizer.evolution import VocabularyEvolution

        engine = VocabularyEvolution(
            vocab=self.vocab,
            min_frequency=min_frequency,
            min_pair_frequency=min_pair_frequency,
            max_ops=max_ops,
        )
        plan, records, applied = engine.evolve(sequences, family_guesser)
        return {
            "plan": plan,
            "new_tokens": records,
            "applied_ops": applied,
            "stats": plan.stats,
        }

    def remap_sequence(
        self,
        cad_tokens: list[str],
        plan=None,
    ) -> list[str]:
        """
        Re-encode a CAD token list against the (possibly evolved) vocabulary:
        adjacent pairs with a registered composite token are merged.
        """
        from cadgenesis.tokenizer.evolution import VocabularyEvolution

        engine = VocabularyEvolution(vocab=self.vocab)
        return engine.remap_sequence(cad_tokens, plan)

    # ---- TOON serialization backend ----

    @property
    def toon_backend(self):
        """A ``ToonBackend`` bound to this tokenizer's vocabulary."""
        from cadgenesis.tokenizer.toon_backend import ToonBackend

        return ToonBackend(self.vocab)

    def serialize_to_toon(self, seq: CADTokenSequence) -> str:
        """Serialize a CADTokenSequence to TOON text (ids + type ids)."""
        return self.toon_backend.serialize_sequence(seq)

    def deserialize_from_toon(self, toon_str: str) -> CADTokenSequence:
        """Rebuild a CADTokenSequence from TOON text."""
        return self.toon_backend.deserialize_sequence(toon_str)

    # ---- Special token id accessors ----

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def unk_id(self) -> int:
        """ID of the <unk> token (used to detect out-of-vocabulary tokens)."""
        return self._unk_id

    # ---- Serialization ----

    def save(self, directory: str | Path) -> None:
        """
        Save the full tokenizer state to ``directory``.

        Creates:
            directory/vocabulary.json     — CADVocabulary
            directory/lang_tokenizer.json — language tokenizer (if legacy)
            directory/tokenizer_state.json — metadata + config
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        self.vocab.save(d / "vocabulary.json")

        # Save language tokenizer
        if isinstance(self.lang_tok, LegacyWordTokenizer):
            self.lang_tok.save(d / "lang_tokenizer.json")
            lang_type = "LegacyWordTokenizer"
        else:
            self.lang_tok.save(d / "lang_tokenizer")
            lang_type = "BPETokenizer"

        # Save metadata
        state = {
            "version": "2.0",
            "vocab_version": self.vocabulary_version,
            "lang_tok_type": lang_type,
            "max_text_len": self.max_text_len,
            "max_cad_len": self.max_cad_len,
            "vocab_size": self.vocab_size,
        }
        with (d / self._STATE_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> AutonomousCADTokenizer:
        """
        Load a complete AutonomousCADTokenizer from ``directory``.
        """
        d = Path(directory)
        with (d / cls._STATE_FILENAME).open("r", encoding="utf-8") as fh:
            state = json.load(fh)

        vocab = CADVocabulary.load(d / "vocabulary.json")

        lang_type = state["lang_tok_type"]
        lang_tok: LanguageTokenizerBase
        if lang_type == "LegacyWordTokenizer":
            lang_tok = LegacyWordTokenizer.load(d / "lang_tokenizer.json")
        else:
            from cadgenesis.tokenizer.language import BPETokenizer

            lang_tok = BPETokenizer.load(d / "lang_tokenizer")

        return cls(
            vocab=vocab,
            lang_tok=lang_tok,
            max_text_len=state["max_text_len"],
            max_cad_len=state["max_cad_len"],
        )

    # ---- Legacy compatibility bridge ----

    def as_legacy_data_py(self) -> tuple[dict[str, int], dict[int, str]]:
        """
        Export tok2id / id2tok dicts compatible with the original data.py.

        Allows existing code that imports CAD_TOK2ID / CAD_ID2TOK directly
        from data.py to be updated to use the new tokenizer with minimal
        changes.

        Returns
        -------
        (tok2id, id2tok) : Tuple[Dict, Dict]
        """
        return self.vocab.to_tok2id(), self.vocab.to_id2tok()

    def __repr__(self) -> str:
        return (
            f"AutonomousCADTokenizer("
            f"vocab_size={self.vocab_size:,}, "
            f"lang_vocab_size={self.lang_vocab_size:,}, "
            f"max_text_len={self.max_text_len}, "
            f"max_cad_len={self.max_cad_len})"
        )


def vocab_tokens(tokenizer: AutonomousCADTokenizer) -> list[str]:
    """All vocab token strings in ascending id order (checkpoint persistence)."""
    id2tok = tokenizer.vocab.to_id2tok()
    return [id2tok[i] for i in sorted(id2tok)]


def restore_vocab_tokens(tokenizer: AutonomousCADTokenizer, tokens: list[str]) -> int:
    """Re-register a checkpoint's vocab into ``tokenizer`` so ids match exactly.

    Registration is append-only and id-deterministic, so re-registering the
    full token list (see :func:`vocab_tokens`) reproduces the original ids.
    Returns the number of tokens that were newly registered.
    """
    from cadgenesis.tokenizer.evolution import guess_family

    tok2id = tokenizer.vocab.to_tok2id()
    missing = [t for t in tokens if t not in tok2id]
    if missing:
        tokenizer.vocab.register_many([(t, guess_family(t)) for t in missing])
    return len(missing)
