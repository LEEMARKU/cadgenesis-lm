"""cadgenesis.tokenizer.serialization
===================================
Standalone serialization helpers for CAD token sequences and tokenizer state.

Thin, dependency-free wrappers over the canonical TOON backend and the
tokenizer's JSON persistence, exposed for reuse outside the tokenizer class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cadgenesis.tokenizer.cad_tokenizer import CADTokenSequence
from cadgenesis.tokenizer.toon_backend import ToonBackend


def serialize_to_toon(seq: CADTokenSequence, vocab) -> str:
    """Serialize a ``CADTokenSequence`` to TOON text via a ToonBackend."""
    return ToonBackend(vocab).serialize_sequence(seq)


def deserialize_from_toon(toon_str: str, vocab) -> CADTokenSequence:
    """Rebuild a ``CADTokenSequence`` from TOON text."""
    return ToonBackend(vocab).deserialize_sequence(toon_str)


def sequence_to_json(seq: CADTokenSequence) -> dict[str, Any]:
    """Dump a ``CADTokenSequence`` to a JSON-serialisable dict."""
    return {
        "text_ids": seq.text_ids,
        "cad_ids": seq.cad_ids,
        "type_ids": seq.type_ids,
        "attention_mask": seq.attention_mask,
        "modality_mask": seq.modality_mask,
        "raw_text": seq.raw_text,
        "raw_cad_tokens": seq.raw_cad_tokens,
    }


def sequence_from_json(data: dict[str, Any]) -> CADTokenSequence:
    """Rebuild a ``CADTokenSequence`` from :func:`sequence_to_json` output."""
    return CADTokenSequence(
        text_ids=list(data.get("text_ids", [])),
        cad_ids=list(data.get("cad_ids", [])),
        type_ids=list(data.get("type_ids", [])),
        attention_mask=list(data.get("attention_mask", [])),
        modality_mask=list(data.get("modality_mask", [])),
        raw_text=str(data.get("raw_text", "")),
        raw_cad_tokens=list(data.get("raw_cad_tokens", [])),
    )


def save_sequences(sequences: list[CADTokenSequence], path: str | Path) -> None:
    """Write a list of sequences as JSON lines to ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for seq in sequences:
            fh.write(json.dumps(sequence_to_json(seq), ensure_ascii=False) + "\n")


def load_sequences(path: str | Path) -> list[CADTokenSequence]:
    """Read JSON-lines ``path`` written by :func:`save_sequences`."""
    p = Path(path)
    sequences: list[CADTokenSequence] = []
    with p.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
        sequences.extend(sequence_from_json(data) for data in lines)
    return sequences


__all__ = [
    "deserialize_from_toon",
    "load_sequences",
    "save_sequences",
    "sequence_from_json",
    "sequence_to_json",
    "serialize_to_toon",
]
