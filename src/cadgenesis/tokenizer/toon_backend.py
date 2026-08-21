"""
cadgenesis.tokenizer.toon_backend
=================================
TOON serialization backend for the Autonomous CAD Tokenizer.

TOON (the standalone `toon.py` / `toon_extended.py` module in the repo root)
remains the *serialization* backend: any tokenizer artifact — a tokenized
CAD sequence or the whole vocabulary — can be flattened to compact, escaped
pipe-delimited text.  The tokenizer itself does the native tokenization;
TOON is only used for interchange / persistence.

This adapter keeps the dependency directional and thin:

    Tokenizer  --(native)-->  ids
    Tokenizer  --(adapter)--> TOON text   (sequence / vocabulary)

Usage
-----
    from cadgenesis.tokenizer import AutonomousCADTokenizer
    tok = AutonomousCADTokenizer.build_mini()
    seq = tok.encode_cad_sequence(["BOX", "NUM_1", "EXTRUDE"])

    text = tok.serialize_to_toon(seq)
    seq2 = tok.deserialize_from_toon(text)

    vocab_text = tok.toon_backend.serialize_vocabulary()
    vocab2 = tok.toon_backend.deserialize_vocabulary(vocab_text)

Complexity
----------
    serialize/deserialize:  O(N)  with N = tokens / records
"""

from __future__ import annotations

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily
from sdk import toon as _toon


class ToonBackend:
    """
    Serializes tokenizer artifacts to / from TOON text.

    Parameters
    ----------
    vocab : CADVocabulary, optional
        Vocabulary used to enrich deserialized sequences with raw token
        strings.  Optional — sequence round-trips only need ids.
    delimiter : str
        Field delimiter for the TOON format (default '|').
    """

    def __init__(self, vocab: CADVocabulary | None = None, delimiter: str = "|") -> None:
        self.vocab = vocab
        self.delimiter = delimiter

    # ----------------------------------------------------------- sequences

    def serialize_sequence(self, seq) -> str:
        """Serialize a CADTokenSequence to TOON text."""
        records = [
            {"id": tok_id, "type": type_id}
            for tok_id, type_id in zip(seq.cad_ids, seq.type_ids, strict=False)
        ]
        return _toon.to_toon(records, delimiter=self.delimiter)

    def deserialize_sequence(self, toon_str: str):
        """Rebuild a CADTokenSequence from TOON text (ids + types)."""
        from cadgenesis.tokenizer.cad_tokenizer import CADTokenSequence

        seq = CADTokenSequence()
        for record in _toon.from_toon(toon_str, delimiter=self.delimiter):
            tok_id = int(record["id"])
            seq.cad_ids.append(tok_id)
            seq.type_ids.append(int(record["type"]))
            seq.attention_mask.append(1)
            if self.vocab is not None and tok_id in self.vocab:
                seq.raw_cad_tokens.append(self.vocab.record_of(tok_id).token_str)
        return seq

    # ---------------------------------------------------------- vocabulary

    def serialize_vocabulary(self, vocab: CADVocabulary | None = None) -> str:
        """
        Serialize a CADVocabulary to TOON text.  Every TokenRecord becomes one
        record: (token, id, family, desc).  Token ids are preserved verbatim.
        """
        vocab = vocab or self.vocab
        if vocab is None:
            raise ValueError("ToonBackend.serialize_vocabulary() needs a vocabulary.")
        records = [
            {
                "token": r.token_str,
                "id": r.token_id,
                "family": r.family.name,
                "desc": r.description,
            }
            for r in vocab
        ]
        return _toon.to_toon(records, delimiter=self.delimiter)

    def serialize_vocabulary_state(self, vocab: CADVocabulary | None = None) -> dict[str, str]:
        """
        Serialize a vocabulary *state* (slot layout + tokens) to two TOON
        strings.  This round-trips custom slot capacities exactly.
        """
        vocab = vocab or self.vocab
        if vocab is None:
            raise ValueError("ToonBackend.serialize_vocabulary_state() needs a vocabulary.")
        slots_records = [
            {"family": family.name, "capacity": capacity}
            for family, capacity in vocab.slot_capacities().items()
        ]
        return {
            "slots": _toon.to_toon(slots_records, delimiter=self.delimiter),
            "tokens": self.serialize_vocabulary(vocab),
        }

    def deserialize_vocabulary(
        self,
        toon_str: str,
        slots: dict[TokenFamily, int] | None = None,
    ) -> CADVocabulary:
        """
        Rebuild a CADVocabulary from TOON text.  Tokens are restored at their
        recorded ids.  Pass ``slots`` to restore a custom slot layout (as
        produced by :meth:`serialize_vocabulary_state`); otherwise the default
        layout is assumed.
        """
        records = _toon.from_toon(toon_str, delimiter=self.delimiter)
        vocab = CADVocabulary(slots=slots) if slots else CADVocabulary()
        for record in records:
            family = TokenFamily[record["family"]]
            vocab.register(
                record["token"],
                family,
                record.get("desc", ""),
                token_id=int(record["id"]),
            )
        return vocab

    def deserialize_vocabulary_state(self, state: dict[str, str]) -> CADVocabulary:
        """
        Rebuild a CADVocabulary from a ``{slots, tokens}`` TOON state dict.
        """
        slots = {
            TokenFamily[record["family"]]: int(record["capacity"])
            for record in _toon.from_toon(state["slots"], delimiter=self.delimiter)
        }
        return self.deserialize_vocabulary(state["tokens"], slots=slots)

    # ---------------------------------------------------------- estimation

    def estimate_tokens(self, text: str, model: str | None = None) -> int:
        """Delegate to TOON's token estimate (tokenizer-agnostic)."""
        return _toon.estimate_tokens(text, model=model)

    # ----------------------------------------------------------- helpers

    def sequence_to_text(self, seq) -> str:
        """Serialize a sequence using its raw token strings (lossless text)."""
        records = [
            {"token": tok, "type": type_id}
            for tok, type_id in zip(seq.raw_cad_tokens, seq.type_ids, strict=False)
        ]
        return _toon.to_toon(records, delimiter=self.delimiter)

    def text_to_tokens(self, toon_str: str) -> list[str]:
        """Deserialize a TOON sequence-text into a raw token list."""
        return [
            str(record["token"]) for record in _toon.from_toon(toon_str, delimiter=self.delimiter)
        ]
