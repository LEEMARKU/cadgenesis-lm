"""
cadgenesis.tokenizer.vocabulary
================================
Extensible multi-modal vocabulary registry for the Autonomous CAD Tokenizer.

Purpose
-------
Manages a flat integer ID space shared across all token families (text,
geometry, feature, constraint, material, assembly, manufacturing, simulation,
numeric).  Each family owns a contiguous ID range and can register tokens
dynamically.  The registry is the single authoritative source of all
``token_string → id`` and ``id → token_string`` mappings.

Architecture
------------
::

    CADVocabulary
    ├── TokenFamily (SPECIAL)       IDs 0-63
    ├── TokenFamily (NUMERIC)       IDs 64-1087
    ├── TokenFamily (GEOMETRY)      IDs 1088-1599
    ├── TokenFamily (FEATURE)       IDs 1600-2111
    ├── TokenFamily (CONSTRAINT)    IDs 2112-2367
    ├── TokenFamily (MATERIAL)      IDs 2368-2623
    ├── TokenFamily (ASSEMBLY)      IDs 2624-2879
    ├── TokenFamily (MANUFACTURING) IDs 2880-3135
    ├── TokenFamily (SIMULATION)    IDs 3136-3391
    └── TokenFamily (LANGUAGE)      IDs 3392-35391 (32000 BPE slots)

Interfaces
----------
    vocab = CADVocabulary.build_default()
    tok_id = vocab["BOX"]                    # → int
    tok_str = vocab[320]                     # → "BOX"
    family = vocab.family_of(320)            # → TokenFamily.GEOMETRY
    type_id = vocab.type_id_of(320)          # → int (for type embedding)
    vocab.register("MY_CUSTOM_FEAT",         # dynamic extension
                   TokenFamily.FEATURE)

Data Structures
---------------
    TokenRecord  — named-tuple holding (token_str, token_id, family, type_id)
    CADVocabulary— central registry; subscriptable by str or int
"""

from __future__ import annotations

import enum
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# TokenFamily — the 9 orthogonal token type families
# ---------------------------------------------------------------------------


class TokenFamily(enum.IntEnum):
    """
    Each family corresponds to one semantic domain.  The integer value is
    also used as the *type-embedding id* fed into the model's type embedding
    layer.  Order must not be changed after models are trained.
    """

    SPECIAL = 0  # <pad>, <bos>, <eos>, <sep>, <mask>, <unk>, …
    NUMERIC = 1  # Quantized parameter values  (NUM_0 … NUM_255)
    GEOMETRY = 2  # Geometric primitives + B-Rep elements
    FEATURE = 3  # CAD feature operations (extrude, fillet, …)
    CONSTRAINT = 4  # Parametric / geometric constraints
    MATERIAL = 5  # Material definitions + physical properties
    ASSEMBLY = 6  # Assembly relationships (mate, align, …)
    MANUFACTURING = 7  # Manufacturing process tokens
    SIMULATION = 8  # Simulation / physics tokens
    LANGUAGE = 9  # BPE text tokens (reserved range)


# ---------------------------------------------------------------------------
# Versioning — vocabulary + serialization schema versions
# ---------------------------------------------------------------------------

#: File-format (schema) version of `CADVocabulary.save/load` payloads.
VOCAB_SCHEMA_VERSION = "1"

#: Semantic version of the default CADGenesis vocabulary definition.  Bumped
#: whenever the canonical token set or the default slot layout changes.
DEFAULT_VOCAB_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# TokenRecord — immutable metadata for a single registered token
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenRecord:
    """Metadata for a single token in the vocabulary."""

    token_str: str
    token_id: int
    family: TokenFamily
    type_id: int  # == family.value — kept explicit for clarity
    description: str = ""
    parts: tuple[str, ...] = ()  # component tokens (composite/merged tokens only)

    def __repr__(self) -> str:
        return f"TokenRecord({self.token_str!r}, id={self.token_id}, family={self.family.name})"


# ---------------------------------------------------------------------------
# _FamilyRange — slot allocation helper
# ---------------------------------------------------------------------------


@dataclass
class _FamilyRange:
    start: int
    capacity: int
    next_free: int = 0

    @property
    def end(self) -> int:
        return self.start + self.capacity

    @property
    def used(self) -> int:
        return self.next_free

    @property
    def remaining(self) -> int:
        return self.capacity - self.next_free

    def allocate(self) -> int:
        if self.next_free >= self.capacity:
            raise OverflowError(
                f"Family range [{self.start}, {self.end}) is full (capacity={self.capacity})."
            )
        slot = self.start + self.next_free
        self.next_free += 1
        return slot


# ---------------------------------------------------------------------------
# CADVocabulary — the central registry
# ---------------------------------------------------------------------------


class CADVocabulary:
    """
    Thread-safe, extensible multi-modal vocabulary registry.

    Token ID space layout (with default slot sizes):

    +-----------------+--------+-----------+
    | Family          | Start  | Capacity  |
    +=================+========+===========+
    | SPECIAL         |      0 |        64 |
    | NUMERIC         |     64 |      1024 |
    | GEOMETRY        |   1088 |       512 |
    | FEATURE         |   1600 |       512 |
    | CONSTRAINT      |   2112 |       256 |
    | MATERIAL        |   2368 |       256 |
    | ASSEMBLY        |   2624 |       256 |
    | MANUFACTURING   |   2880 |       256 |
    | SIMULATION      |   3136 |       256 |
    | LANGUAGE        |   3392 |    32 000 |
    +-----------------+--------+-----------+
    | Total                    |    34 624 |
    +-----------------+--------+-----------+
    """

    # Default slot capacities (can be overridden in __init__)
    DEFAULT_SLOTS: dict[TokenFamily, int] = {
        TokenFamily.SPECIAL: 64,
        TokenFamily.NUMERIC: 1024,
        TokenFamily.GEOMETRY: 512,
        TokenFamily.FEATURE: 512,
        TokenFamily.CONSTRAINT: 256,
        TokenFamily.MATERIAL: 256,
        TokenFamily.ASSEMBLY: 256,
        TokenFamily.MANUFACTURING: 256,
        TokenFamily.SIMULATION: 256,
        TokenFamily.LANGUAGE: 32_000,
    }

    def __init__(
        self,
        slots: dict[TokenFamily, int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        _slots = {**self.DEFAULT_SLOTS, **(slots or {})}

        # Build contiguous ranges per family
        self._ranges: dict[TokenFamily, _FamilyRange] = {}
        cursor = 0
        for family in TokenFamily:
            cap = _slots.get(family, 0)
            self._ranges[family] = _FamilyRange(start=cursor, capacity=cap)
            cursor += cap

        self._total_capacity: int = cursor

        # Bidirectional lookups
        self._str2record: dict[str, TokenRecord] = {}
        self._id2record: dict[int, TokenRecord] = {}

        #: Semantic vocabulary version (see DEFAULT_VOCAB_VERSION).
        self.version: str = DEFAULT_VOCAB_VERSION

        #: Old→new id mapping / dropped ids from the most recent migration.
        self._migration_id_mapping: dict[int, int] | None = None
        self._migration_unmapped: list[int] = []

    # ---- Core registration ----

    def register(
        self,
        token_str: str,
        family: TokenFamily,
        description: str = "",
        token_id: int | None = None,
        parts: tuple[str, ...] = (),
    ) -> TokenRecord:
        """
        Register a new token.  Returns the TokenRecord (including assigned id).

        When ``token_id`` is given the token is pinned to that exact id (used
        by the TOON serialization backend to restore vocabularies verbatim).
        Otherwise the family's next free slot is allocated.

        ``parts`` records the component tokens of a composite (merged) token
        so that compression can be reversed losslessly via ``expand_token``.

        Raises:
            KeyError   — if token_str is already registered.
            OverflowError — if the family's slot range is exhausted.
        """
        with self._lock:
            if token_str in self._str2record:
                raise KeyError(f"Token {token_str!r} is already registered.")
            if token_id is None:
                tok_id = self._ranges[family].allocate()
            else:
                rng = self._ranges[family]
                if not (rng.start <= token_id < rng.end):
                    raise ValueError(
                        f"token_id {token_id} is outside the {family.name} "
                        f"range [{rng.start}, {rng.end})."
                    )
                if token_id in self._id2record:
                    raise KeyError(f"Token id {token_id} is already registered.")
                tok_id = token_id
                if tok_id >= rng.start + rng.next_free:
                    rng.next_free = tok_id - rng.start + 1
            record = TokenRecord(
                token_str=token_str,
                token_id=tok_id,
                family=family,
                type_id=family.value,
                description=description,
                parts=tuple(parts),
            )
            self._str2record[token_str] = record
            self._id2record[tok_id] = record
            return record

    def register_many(
        self,
        tokens: list[tuple[str, TokenFamily]],
        descriptions: list[str] | None = None,
    ) -> list[TokenRecord]:
        """Batch registration; returns list of TokenRecords in order."""
        descs = descriptions or [""] * len(tokens)
        return [
            self.register(tok, fam, desc) for (tok, fam), desc in zip(tokens, descs, strict=False)
        ]

    # ---- Dynamic growth / evolution operations ----

    def remove_token(self, key: str | int) -> TokenRecord:
        """
        Remove a token from the registry.  If the removed token was the last
        allocated slot of its family, that slot is reclaimed.
        """
        with self._lock:
            record = self.record_of(key)
            del self._str2record[record.token_str]
            del self._id2record[record.token_id]
            rng = self._ranges[record.family]
            if record.token_id == rng.start + rng.next_free - 1:
                rng.next_free -= 1
            return record

    def trim_unused(self, predicate) -> list[TokenRecord]:
        """
        Remove every token for which ``predicate(record)`` is True.
        Returns the removed TokenRecords.
        """
        removed = [r for r in self if predicate(r)]
        for record in removed:
            self.remove_token(record.token_id)
        return removed

    def merge_tokens(
        self,
        parts: list[str],
        merged_str: str,
        family: TokenFamily,
        description: str = "",
    ) -> TokenRecord:
        """
        Register a composite token ``merged_str`` formed from ``parts``
        (which must already be registered).  The parts are kept — the merged
        token is an additional vocabulary entry used to compress frequent
        co-occurring sequences.
        """
        for part in parts:
            if part not in self._str2record:
                raise KeyError(f"Merge part {part!r} is not registered.")
        return self.register(merged_str, family, description, parts=tuple(parts))

    def expand_token(self, token_str: str) -> list[str]:
        """
        Losslessly expand a composite (merged) token back into its component
        token strings.  Non-composite tokens return ``[token_str]``.  Parts are
        expanded recursively, so nested merges flatten fully.
        """
        record = self._str2record.get(token_str)
        if record is None or not record.parts:
            return [token_str]
        flattened: list[str] = []
        for part in record.parts:
            flattened.extend(self.expand_token(part))
        return flattened

    def split_token(
        self,
        token: str,
        part_strs: list[str],
        family: TokenFamily,
        descriptions: list[str] | None = None,
    ) -> list[TokenRecord]:
        """
        Register the ``part_strs`` pieces of ``token``.  Parts that already
        exist are returned as-is.  Returns the list of part TokenRecords.
        """
        if token not in self._str2record:
            raise KeyError(f"Token {token!r} is not registered.")
        descs = descriptions or [""] * len(part_strs)
        records = []
        for part, desc in zip(part_strs, descs, strict=False):
            if part in self._str2record:
                records.append(self._str2record[part])
            else:
                records.append(self.register(part, family, desc))
        return records

    def remaining_slots(self, family: TokenFamily) -> int:
        """Number of unallocated slots left in a family's range."""
        return self._ranges[family].remaining

    # ---- Subscript access ----

    def __getitem__(self, key: str | int) -> int | str:
        """
        vocab["BOX"]  → int token id
        vocab[320]    → str token string
        """
        if isinstance(key, str):
            return self._str2record[key].token_id
        if isinstance(key, int):
            return self._id2record[key].token_str
        raise TypeError(f"Key must be str or int, got {type(key).__name__}.")

    def __contains__(self, key: str | int) -> bool:
        if isinstance(key, str):
            return key in self._str2record
        if isinstance(key, int):
            return key in self._id2record
        return False

    def __len__(self) -> int:
        return len(self._str2record)

    def __iter__(self) -> Iterator[TokenRecord]:
        """Iterates records in token_id order."""
        return iter(sorted(self._id2record.values(), key=lambda r: r.token_id))

    # ---- Metadata accessors ----

    def record_of(self, key: str | int) -> TokenRecord:
        """Return full TokenRecord for a token string or id."""
        if isinstance(key, str):
            return self._str2record[key]
        return self._id2record[key]

    def family_of(self, key: str | int) -> TokenFamily:
        return self.record_of(key).family

    def type_id_of(self, key: str | int) -> int:
        """Returns the integer family value (type embedding input)."""
        return self.record_of(key).type_id

    def tokens_in_family(self, family: TokenFamily) -> list[TokenRecord]:
        """All registered tokens belonging to a given family, id-sorted."""
        return sorted(
            [r for r in self._str2record.values() if r.family == family],
            key=lambda r: r.token_id,
        )

    def family_range(self, family: TokenFamily) -> tuple[int, int]:
        """Returns (start_id, end_id_exclusive) for a family's slot range."""
        r = self._ranges[family]
        return r.start, r.end

    def slot_capacities(self) -> dict[TokenFamily, int]:
        """Return the configured slot capacity per family (layout definition)."""
        return {family: rng.capacity for family, rng in self._ranges.items()}

    # ---- Capacity stats ----

    def stats(self) -> dict[str, dict]:
        result = {}
        for family, rng in self._ranges.items():
            result[family.name] = {
                "start": rng.start,
                "end": rng.end,
                "capacity": rng.capacity,
                "used": rng.used,
                "remaining": rng.remaining,
            }
        result["TOTAL"] = {
            "capacity": self._total_capacity,
            "used": len(self._str2record),
        }
        return result

    # ---- Layout migration ----

    def migrate_layout(
        self,
        new_slots: dict[TokenFamily, int],
        new_version: str | None = None,
    ) -> CADVocabulary:
        """
        Rebuild the vocabulary under a new slot layout, preserving token IDs
        wherever they still fit inside their family's new range.

        Tokens whose old ID falls outside the new range are re-registered into
        the first free slot of their family (the old→new id mapping is
        recorded).  Tokens that cannot be re-registered because their family's
        new capacity is exhausted are dropped (reported via
        ``MigrationResult.unmapped_ids`` — see ``cadgenesis.tokenizer.versioning``).

        Returns a *new* CADVocabulary; the receiver is not modified.
        """
        new_vocab = CADVocabulary(slots=new_slots)
        if new_version is not None:
            new_vocab.version = new_version

        id_mapping: dict[int, int] = {}
        unmapped: list[int] = []
        for record in self:
            start, end = new_vocab.family_range(record.family)
            if start <= record.token_id < end and record.token_id not in new_vocab._id2record:
                new_vocab.register(
                    record.token_str,
                    record.family,
                    record.description,
                    token_id=record.token_id,
                    parts=record.parts,
                )
                id_mapping[record.token_id] = record.token_id
            else:
                try:
                    new_rec = new_vocab.register(
                        record.token_str,
                        record.family,
                        record.description,
                        parts=record.parts,
                    )
                    id_mapping[record.token_id] = new_rec.token_id
                except OverflowError:
                    unmapped.append(record.token_id)
        new_vocab._migration_id_mapping = id_mapping
        new_vocab._migration_unmapped = unmapped
        return new_vocab

    @property
    def last_migration(self) -> tuple[dict[int, int], list[int]] | None:
        """
        Returns ``(id_mapping, unmapped_ids)`` from the most recent
        ``migrate_layout`` call on this vocabulary, or ``None`` if the
        vocabulary was not produced by a migration.
        """
        mapping = getattr(self, "_migration_id_mapping", None)
        if mapping is None:
            return None
        return mapping, getattr(self, "_migration_unmapped", [])

    # ---- Serialization ----

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON (token_str → token_id mapping + metadata).

        The payload carries ``vocab_version`` (semantic vocabulary version),
        ``schema_version`` (file-format version) and, per token, the composite
        ``parts`` so compression info survives round-trips.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "vocab_version": self.version,
            "schema_version": VOCAB_SCHEMA_VERSION,
            "tokens": [
                {
                    "token_str": r.token_str,
                    "token_id": r.token_id,
                    "family": r.family.name,
                    "type_id": r.type_id,
                    "description": r.description,
                    "parts": list(r.parts),
                }
                for r in self
            ],
            "slot_capacities": {fam.name: rng.capacity for fam, rng in self._ranges.items()},
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CADVocabulary:
        """Reconstruct a CADVocabulary from a saved JSON file.

        Backward compatible: files written by older versions that lack the
        ``vocab_version`` / ``parts`` fields load with sensible defaults.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        slots = {TokenFamily[name]: cap for name, cap in payload["slot_capacities"].items()}
        vocab = cls(slots=slots)
        vocab.version = payload.get("vocab_version", DEFAULT_VOCAB_VERSION)
        for entry in payload["tokens"]:
            family = TokenFamily[entry["family"]]
            vocab.register(
                entry["token_str"],
                family,
                entry.get("description", ""),
                parts=tuple(entry.get("parts", []) or []),
            )
        return vocab

    # ---- Legacy compatibility helpers ----

    def to_tok2id(self) -> dict[str, int]:
        """Produce a flat tok→id dict compatible with existing data.py usage."""
        return {r.token_str: r.token_id for r in self}

    def to_id2tok(self) -> dict[int, str]:
        """Produce a flat id→tok dict compatible with existing data.py usage."""
        return {r.token_id: r.token_str for r in self}

    # ---- Factory ----

    @classmethod
    def build_default(cls) -> CADVocabulary:
        """
        Build the full default CADGenesis-LM v2.0 vocabulary.

        Delegates to each sub-tokenizer's `populate()` factory method so
        this class stays thin and sub-tokenizers own their token lists.
        """
        # Import here to avoid circular imports
        from cadgenesis.tokenizer.assembly import AssemblyTokenizer
        from cadgenesis.tokenizer.constraint import ConstraintTokenizer
        from cadgenesis.tokenizer.feature import FeatureTokenizer
        from cadgenesis.tokenizer.geometry import GeometryTokenizer
        from cadgenesis.tokenizer.manufacturing import ManufacturingTokenizer
        from cadgenesis.tokenizer.material import MaterialTokenizer
        from cadgenesis.tokenizer.numeric import NumericTokenizer
        from cadgenesis.tokenizer.simulation import SimulationTokenizer

        vocab = cls()

        # 1. Special tokens first — their IDs are foundational
        _register_special_tokens(vocab)

        # 2. Numeric quantization tokens
        NumericTokenizer.populate(vocab)

        # 3. Domain-specific token families
        GeometryTokenizer.populate(vocab)
        FeatureTokenizer.populate(vocab)
        ConstraintTokenizer.populate(vocab)
        MaterialTokenizer.populate(vocab)
        AssemblyTokenizer.populate(vocab)
        ManufacturingTokenizer.populate(vocab)
        SimulationTokenizer.populate(vocab)

        # NOTE: LANGUAGE slots are populated by the BPE language tokenizer
        # at training time (or loaded from a pretrained vocab file).

        return vocab


# ---------------------------------------------------------------------------
# Special token registration (IDs 0-63 reserved)
# ---------------------------------------------------------------------------

# Canonical special token strings — accessed as module-level constants.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"
CLS_TOKEN = "<cls>"

# Domain-specific control tokens
CAD_START_TOKEN = "<cad_start>"
CAD_END_TOKEN = "<cad_end>"
CONSTRAINT_START_TOKEN = "<constraint_start>"
CONSTRAINT_END_TOKEN = "<constraint_end>"
ASSEMBLY_START_TOKEN = "<assembly_start>"
ASSEMBLY_END_TOKEN = "<assembly_end>"
MATERIAL_START_TOKEN = "<material_start>"
MATERIAL_END_TOKEN = "<material_end>"
MANUF_START_TOKEN = "<manufacturing_start>"
MANUF_END_TOKEN = "<manufacturing_end>"
SIM_START_TOKEN = "<sim_start>"
SIM_END_TOKEN = "<sim_end>"
AGENT_TOKEN = "<agent>"
MEMORY_TOKEN = "<memory>"
THINK_TOKEN = "<think>"
ANSWER_TOKEN = "<answer>"


_SPECIAL_TOKENS: list[tuple[str, str]] = [
    (PAD_TOKEN, "Padding token (id=0, always first)"),
    (BOS_TOKEN, "Beginning of sequence"),
    (EOS_TOKEN, "End of sequence"),
    (UNK_TOKEN, "Unknown / out-of-vocabulary token"),
    (SEP_TOKEN, "Separator between modalities"),
    (MASK_TOKEN, "Masked position (for MLM-style training)"),
    (CLS_TOKEN, "Classification / pooling token"),
    (CAD_START_TOKEN, "Start of a CAD construction sequence"),
    (CAD_END_TOKEN, "End of a CAD construction sequence"),
    (CONSTRAINT_START_TOKEN, "Start of parametric constraint block"),
    (CONSTRAINT_END_TOKEN, "End of parametric constraint block"),
    (ASSEMBLY_START_TOKEN, "Start of assembly definition block"),
    (ASSEMBLY_END_TOKEN, "End of assembly definition block"),
    (MATERIAL_START_TOKEN, "Start of material definition block"),
    (MATERIAL_END_TOKEN, "End of material definition block"),
    (MANUF_START_TOKEN, "Start of manufacturing specification block"),
    (MANUF_END_TOKEN, "End of manufacturing specification block"),
    (SIM_START_TOKEN, "Start of simulation parameters block"),
    (SIM_END_TOKEN, "End of simulation parameters block"),
    (AGENT_TOKEN, "Inter-agent communication marker"),
    (MEMORY_TOKEN, "Memory read/write marker"),
    (THINK_TOKEN, "Internal reasoning chain start"),
    (ANSWER_TOKEN, "Final answer / output start"),
]


def _register_special_tokens(vocab: CADVocabulary) -> None:
    for token_str, desc in _SPECIAL_TOKENS:
        vocab.register(token_str, TokenFamily.SPECIAL, desc)
