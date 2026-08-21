"""cadgenesis.datasets.cad_jsonl
===============================
Real CAD dataset pipeline (P0 modernization).

Loads ``(text, cad-token-sequence)`` pairs from JSONL files (one JSON object
per line) and provides dependency-free **MinHash deduplication** so training
is not dominated by repeated files (the modern data-curation standard).

Supported record shapes::

    {"text": "create a steel box 50mm wide",
     "cad": ["SKETCH_RECT", "NUM_50", ...]}                       # token strings
    {"text": "...", "cad_ids": [10, 20, 30]}                      # token ids
    {"src": [1, 2, 3], "tgt": [10, 20, 30]}                       # raw id pairs

The dataset yields raw ``(src_ids, tgt_ids)`` pairs suitable for the trainer's
``MultiModalCADDataset`` / packed collate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def _stable_hash(text: str) -> int:
    """Deterministic 64-bit hash (Python's builtin hash is salted per-run)."""
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")


def _shingles(seq: Sequence[str], k: int) -> list[str]:
    """Character k-shingles of a token sequence (for similarity hashing)."""
    if len(seq) <= k:
        return ["\x00".join(seq)]
    return ["\x00".join(seq[i : i + k]) for i in range(len(seq) - k + 1)]


def minhash_signature(seq: Sequence[str], num_hashes: int = 16) -> list[int]:
    """
    MinHash signature of a token sequence: the minimum of ``num_hashes``
    deterministic hashes of its k-shingles.  Two sequences with similar
    content share many signature entries regardless of length.
    """
    k = 4
    sig = [2**63 - 1] * num_hashes
    for shingle in _shingles(list(seq), k):
        for i in range(num_hashes):
            # Different hash functions via salted MD5 mixing.
            mixed = _stable_hash(f"{i}:{shingle}")
            if mixed < sig[i]:
                sig[i] = mixed
    return sig


def minhash_dedup(
    records: list[dict[str, Any]],
    num_hashes: int = 16,
    threshold: float = 0.35,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Deduplicate records whose MinHash signatures are near-duplicates.

    Records are kept in order; a record is dropped when it shares at least
    ``threshold`` fraction of signature entries with a previously kept record.

    Parameters
    ----------
    records : list[dict]
        Records with a ``cad`` (token strings) or ``cad_ids``/``tgt`` field.
    num_hashes : int
        Signature length (more hashes = more accurate, slower).
    threshold : float
        Jaccard-similarity threshold in [0, 1]; drop a record whose best
        signature overlap with a kept record exceeds this.

    Returns the deduplicated list (original order preserved).
    """
    if not records:
        return records
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    kept: list[dict[str, Any]] = []
    kept_sigs: list[list[int]] = []
    dropped = 0
    for record in records:
        seq = _record_tokens(record)
        if not seq:
            kept.append(record)
            continue
        sig = minhash_signature(seq, num_hashes=num_hashes)
        is_dup = False
        for other_sig in kept_sigs:
            overlap = sum(1 for a, b in zip(sig, other_sig, strict=True) if a == b) / num_hashes
            if overlap >= threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(record)
        kept_sigs.append(sig)

    if progress and dropped:
        print(f"[cad_jsonl] MinHash dedup: kept {len(kept)} / {len(records)} (dropped {dropped})")
    return kept


def _record_tokens(record: dict[str, Any]) -> list[str]:
    """Extract token strings from any supported record shape (best-effort)."""
    if isinstance(record.get("cad"), list) and record["cad"] and isinstance(record["cad"][0], str):
        return record["cad"]
    if isinstance(record.get("tgt"), list) and record["tgt"]:
        return [str(t) for t in record["tgt"]]
    if isinstance(record.get("cad_ids"), list) and record["cad_ids"]:
        return [str(t) for t in record["cad_ids"]]
    return []


def load_jsonl(path: str | Path, max_records: int | None = None) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts (skipping malformed lines)."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
                if max_records is not None and len(records) >= max_records:
                    break
    return records


class CADJsonlDataset(Dataset):
    """
    Dataset over JSONL ``(text, cad)`` pairs, yielding raw token-id pairs.

    Parameters
    ----------
    path : str | Path
        JSONL file with ``text`` + ``cad`` (token strings) records.
    tokenizer : object
        Must implement ``encode_text(str) -> list[int]`` and
        ``encode_cad_sequence(list[str], add_bos=False, add_eos=False) ->
        object with ``cad_ids`` attribute`` (the ``AutonomousCADTokenizer``
        contract).
    max_records : int, optional
        Only load the first ``max_records`` lines.
    dedup : bool
        Apply MinHash near-duplicate removal on load.
    dedup_threshold : float
        Similarity threshold for ``dedup``.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_records: int | None = None,
        dedup: bool = False,
        dedup_threshold: float = 0.35,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"dataset not found: {self.path}")
        records = load_jsonl(self.path, max_records=max_records)
        if dedup:
            records = minhash_dedup(records, threshold=dedup_threshold)
        self.tokenizer = tokenizer
        self.pairs: list[tuple[list[int], list[int]]] = []
        for record in records:
            text = record.get("text")
            if text is None:
                continue
            seq = self.tokenizer.encode_cad_sequence(record["cad"], add_bos=False, add_eos=False)
            self.pairs.append((self.tokenizer.encode_text(text), list(seq.cad_ids)))
        if not self.pairs:
            raise ValueError(f"no usable (text, cad) pairs in {self.path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.pairs[idx]

    def iter_text_cad(self) -> Iterator[tuple[list[int], list[str]]]:
        """Yield ``(encoded_text_ids, cad_token_strings)`` pairs (for stats/EDA)."""
        for text, ids in self.pairs:
            yield text, self.tokenizer.decode_cad_sequence(ids)


def split_records(
    records: list[dict[str, Any]], valid_fraction: float = 0.05, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train/validation split of loaded records."""
    import random

    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1).")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    # Always keep at least one record for training (tiny datasets).
    n_valid = min(
        max(1, round(len(shuffled) * valid_fraction)),
        max(0, len(shuffled) - 1),
    )
    return shuffled[n_valid:], shuffled[:n_valid]


__all__ = [
    "CADJsonlDataset",
    "load_jsonl",
    "minhash_dedup",
    "minhash_signature",
    "split_records",
]
