"""cadgenesis.rag.rag_engine
==========================
Retrieval-augmented generation for CAD programs (M6).

Bridges the semantic memory layer (:mod:`cadgenesis.memory`) with the trained
model (:class:`cadgenesis.inference.engine.CADInferenceEngine`):

    user prompt -> MemoryRetrieval (top-k exemplars) -> augmented prompt
                 -> CADInferenceEngine -> program

The corpus is any JSONL of ``{"text", "cad", ...}`` records (the M3
curriculum files qualify); records are indexed into a
:class:`~cadgenesis.memory.memory_common.MemoryStore` keyed by their CAD-IR
``program_id``.

Augmentation is honest about the tokenizer's text window: the prompt builder
respects ``tokenizer.max_text_len`` (the mini model used by M4 accepts 32
tokens), so the query is augmented with at most one exemplar line per
retrieved record and the result is truncated to the model's real window.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cadgenesis.ir import parse_program
from cadgenesis.memory.memory_common import MemoryEntry, MemoryStore
from cadgenesis.memory.retrieval import MemoryRetrieval, RetrievalResult

#: Default exemplars injected into the augmented prompt.
DEFAULT_TOP_K = 1

#: Searchable text prefix per retrieved exemplar.
_REFERENCE_PREFIX = "reference:"


def _template_of(cad: Iterable[str]) -> tuple[str, ...]:
    """CAD operation-kind sequence (e.g. ``PRIM_BOX -> FEAT_EXTRUDE``)."""
    try:
        return tuple(op.kind for op in parse_program(list(cad)).steps)
    except Exception:
        return ()


class CADRAGEngine:
    """Index, retrieve and generate with retrieval augmentation."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        retrieval: MemoryRetrieval | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.store = store or MemoryStore(name="cad-rag", capacity=1_000_000)
        self.retrieval = retrieval or MemoryRetrieval()
        self.retrieval.register(self.store)
        self.top_k = top_k

    # ----------------------------------------------------------------- index

    def index_record(self, record: dict[str, Any]) -> MemoryEntry:
        """Index one ``{"text", "cad"}`` record (key = CAD-IR program_id)."""
        cad = list(record.get("cad") or [])
        key = str(record.get("program_id") or "")
        if not key and cad:
            key = parse_program(cad).program_id
        if not key:
            key = f"rec-{len(self.store)}"
        content = {"text": str(record.get("text", "")), "cad": cad}
        metadata: dict[str, Any] = {}
        for meta_key in ("type", "score", "quality"):
            if meta_key in record:
                metadata[meta_key] = record[meta_key]
        importance = float(record.get("score", 1.0)) if record.get("score") is not None else 1.0
        return self.store.add(key, content, importance=importance, metadata=metadata)

    def index_records(self, records: Iterable[dict[str, Any]]) -> int:
        """Index many records; returns the number indexed."""
        count = 0
        for record in records:
            self.index_record(record)
            count += 1
        return count

    def index_jsonl(self, path: str | Path) -> int:
        """Index a JSONL corpus file (one record per line)."""
        count = 0
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                self.index_record(json.loads(line))
                count += 1
        return count

    # -------------------------------------------------------------- retrieve

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Ranked retrieval over the indexed corpus."""
        return self.retrieval.retrieve(query, top_k=top_k or self.top_k)

    def precision_at_k(self, query: str, expected_key: str, k: int | None = None) -> float:
        """1.0 when the expected record key is inside the top-k hits."""
        result = self.retrieve(query, top_k=k or self.top_k)
        return 1.0 if any(hit.entry.key == expected_key for hit in result.hits) else 0.0

    def template_precision_at_k(
        self, query: str, cad: Iterable[str], k: int | None = None
    ) -> float:
        """1.0 when a top-k hit shares the query's CAD operation template.

        The dataset splits are leakage-free (a val ``program_id`` never
        exists in the train corpus), so exact-key recall is meaningless
        across splits; the operation-kind sequence (e.g. ``PRIM_BOX ->
        FEAT_EXTRUDE``) is the retrievable unit.
        """
        expected = _template_of(cad)
        if not expected:
            return 0.0
        result = self.retrieve(query, top_k=k or self.top_k)
        for hit in result.hits:
            content = hit.entry.content
            if isinstance(content, dict) and _template_of(content.get("cad") or []) == expected:
                return 1.0
        return 0.0

    # ---------------------------------------------------------------- augment

    def augmented_prompt(self, query: str, top_k: int | None = None) -> str:
        """Query plus retrieved exemplar lines, sized to the model window."""
        result = self.retrieve(query, top_k=top_k or self.top_k)
        parts = [query]
        for hit in result.hits:
            text = hit.entry.content.get("text") if isinstance(hit.entry.content, dict) else None
            if text:
                parts.append(f"{_REFERENCE_PREFIX} {text}")
        return " ".join(parts)

    # --------------------------------------------------------------- generate

    def generate(
        self,
        engine: Any,
        query: str,
        top_k: int | None = None,
        max_len: int = 24,
        **decode_kwargs: Any,
    ) -> dict[str, Any]:
        """Retrieve, augment and generate a program.

        ``engine`` is a :class:`CADInferenceEngine`-compatible object with
        ``greedy(text, max_len, **kwargs)`` / ``sample(...)``; decode kwargs
        (``temperature``, ``top_k``, ``use_cache``) pass through.
        """
        augmented = self.augmented_prompt(query, top_k=top_k or self.top_k)
        t0 = time.perf_counter()
        if "temperature" in decode_kwargs:
            result = engine.sample(augmented, max_len=max_len, **decode_kwargs)
        else:
            result = engine.greedy(augmented, max_len=max_len, **decode_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": query,
            "augmented_prompt": augmented,
            "tokens": list(result.tokens),
            "ids": list(result.ids),
            "stopped_on_eos": result.stopped_on_eos,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    # -------------------------------------------------------------- benchmark

    def benchmark_retrieval(
        self,
        queries: Iterable[tuple[str, str]],
        k: int | None = None,
        template: bool = True,
    ) -> dict[str, Any]:
        """Mean precision@k over ``(query, cad_or_key)`` pairs.

        With ``template=True`` (default) hits are scored by operation-kind
        template match; with ``template=False`` by exact record key.
        """
        pairs = list(queries)
        hits = 0.0
        total_ms = 0.0
        for query, target in pairs:
            t0 = time.perf_counter()
            if template:
                hits += self.template_precision_at_k(query, target, k=k or self.top_k)
            else:
                hits += self.precision_at_k(query, target, k=k or self.top_k)
            total_ms += (time.perf_counter() - t0) * 1000.0
        n = len(pairs)
        return {
            "n": n,
            "k": k or self.top_k,
            "precision_at_k": hits / n if n else 0.0,
            "mean_ms": round(total_ms / n, 3) if n else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "store": self.store.summary(),
            "top_k": self.top_k,
            "pool_names": self.retrieval.pool_names,
        }


__all__ = ["DEFAULT_TOP_K", "CADRAGEngine"]
