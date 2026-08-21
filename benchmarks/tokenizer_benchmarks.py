"""
benchmarks/tokenizer_benchmarks.py
====================================
Benchmarks for the CAD Tokenizer subsystem: encoding, decoding, vocabulary
build, corpus statistics and compression.

Run:
    python benchmarks/tokenizer_benchmarks.py [--reps 20] [--seq-len 512]

Measured on CPU (torch not required).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer


def _timeit(fn: Callable[[], object], reps: int) -> float:
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    reps = args.reps
    n = args.seq_len
    tokens: list[str] = (["BOX", "CYLINDER", "SPHERE"] * (n // 3 + 1))[:n]

    print(f"benchmark: tokenizer (CPU) — reps={reps}, seq_len={n}\n")

    t = time.perf_counter()
    tok = AutonomousCADTokenizer.build_mini()
    build_ms = (time.perf_counter() - t) * 1e3
    print(f"build_mini vocabulary:          {build_ms:9.3f} ms")

    t = time.perf_counter()
    tok = AutonomousCADTokenizer.build()
    build_full_ms = (time.perf_counter() - t) * 1e3
    print(f"build_default vocabulary:       {build_full_ms:9.3f} ms  ({tok.vocab_size:,} tokens)")

    tok.encode_cad_sequence(tokens[:64])
    per_encode = _timeit(lambda: tok.encode_cad_sequence(tokens[:64]), reps) * 1e3
    print(f"encode_cad_sequence (64 tok):   {per_encode:9.3f} ms/seq")

    ids = tok.encode_cad_sequence(tokens).cad_ids
    per_decode = _timeit(lambda: tok.decode_cad_sequence(ids), reps) * 1e3
    print(f"decode_cad_sequence ({n} tok):   {per_decode:9.3f} ms/seq")

    tok.token_statistics([tok.encode_cad_sequence(tokens)])
    per_stats = _timeit(lambda: tok.token_statistics([tok.encode_cad_sequence(tokens)]), reps) * 1e3
    print(f"token_statistics (1 seq):       {per_stats:9.3f} ms")

    # Register a composite token for the frequent pair, then re-measure
    from cadgenesis.tokenizer.vocabulary import TokenFamily

    tok.vocab.register("BOX_CYLINDER", TokenFamily.GEOMETRY, "merged", parts=("BOX", "CYLINDER"))
    per_compress = _timeit(lambda: tok.compress_sequence(tokens), reps) * 1e3
    comp_tokens, comp_ratio = tok.compress_sequence(tokens)
    lossless = tok.expand_sequence(comp_tokens)
    print(f"compress_sequence ({n} tok):     {per_compress:9.3f} ms")
    print(f"  corpus compression_ratio:     {comp_ratio:.4f}  (lossless={lossless == tokens})")

    migrated = tok.migrate_vocabulary(dict(tok.vocab.slot_capacities()))
    per_migrate = (
        _timeit(lambda: tok.migrate_vocabulary(dict(tok.vocab.slot_capacities())), 3) * 1e3
    )
    print(
        f"migrate_vocabulary (same):      {per_migrate:9.3f} ms  "
        f"({migrated.preserved_ids:,} ids preserved)"
    )


if __name__ == "__main__":
    main()
