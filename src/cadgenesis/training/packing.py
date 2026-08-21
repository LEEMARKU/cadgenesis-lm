"""
cadgenesis.training.packing
===========================
Sequence packing utilities (P0 modernization).

Packing concatenates multiple variable-length ``(text, CAD)`` samples into a
single fixed-length row, dramatically increasing token utilisation compared
with pad-to-max-length collation.  Cross-sample attention is blocked with
block-diagonal masks, so a packed row is *exactly* equivalent to processing
the samples separately:

* Encoder row:      ``[src_1 EOS src_2 EOS ...]``
* Decoder row:      ``[BOS tgt_1 EOS BOS tgt_2 EOS ...]``
* ``src_attn_mask``     blocks encoder self-attention across samples
* ``tgt_attn_mask``     blocks decoder self-attention across samples and is
                        causal within each sample
* ``cross_attn_mask``   blocks decoder->encoder attention across samples

All masks are returned in ``(B, 1, T, T)`` / ``(B, 1, T, S)`` shape so they
broadcast against ``(B, H, T, T)`` attention scores.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def _trim(src: Sequence[int], tgt: Sequence[int], max_src_len: int, max_tgt_len: int):
    """Truncate a sample so it always fits inside a packed row."""
    src = list(src)[: max(1, max_src_len - 1)]
    tgt = list(tgt)[: max(1, max_tgt_len - 2)]
    return src, tgt


def pack_batch(
    samples: Sequence[tuple[Sequence[int], Sequence[int]]],
    max_src_len: int,
    max_tgt_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int = 0,
    seed: int | None = None,
) -> dict:
    """
    Greedy bin-pack ``(src_ids, tgt_ids)`` samples into fixed-length rows.

    Samples are packed in order (optionally shuffled with ``seed``); a sample
    that does not fit in the current row starts a new row.  Every sample is
    guaranteed to fit after truncation.

    Returns
    -------
    dict with keys:
        src             (B, max_src_len) long
        src_pad_mask    (B, max_src_len) bool   — True where padded
        tgt             (B, max_tgt_len) long   — includes BOS/EOS per sample
        loss_mask       (B, max_tgt_len - 1) bool — True where a target token
                        should be scored (i.e. non-padding)
        src_attn_mask   (B, 1, max_src_len, max_src_len) block mask
        tgt_attn_mask   (B, 1, max_tgt_len, max_tgt_len) block causal mask
        cross_attn_mask (B, 1, max_tgt_len, max_src_len) decoder->encoder mask
        n_packed        (B,) int — samples per row (statistics)
    """
    if max_src_len < 2 or max_tgt_len < 2:
        raise ValueError("max_src_len / max_tgt_len must be >= 2 for packing.")
    if len(samples) == 0:
        raise ValueError("samples must not be empty.")

    order = list(range(len(samples)))
    if seed is not None:
        random.Random(seed).shuffle(order)

    rows_src: list[list[int]] = []
    rows_tgt: list[list[int]] = []
    src_bounds: list[list[tuple[int, int]]] = []
    tgt_bounds: list[list[tuple[int, int]]] = []
    packed_counts: list[int] = []

    cur_src: list[int] = []
    cur_tgt: list[int] = []
    cur_src_bounds: list[tuple[int, int]] = []
    cur_tgt_bounds: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal cur_src, cur_tgt, cur_src_bounds, cur_tgt_bounds
        if cur_src:
            rows_src.append(cur_src)
            rows_tgt.append(cur_tgt)
            src_bounds.append(cur_src_bounds)
            tgt_bounds.append(cur_tgt_bounds)
            packed_counts.append(len(cur_src_bounds))
        cur_src, cur_tgt = [], []
        cur_src_bounds, cur_tgt_bounds = [], []

    for i in order:
        src_ids, tgt_ids = _trim(samples[i][0], samples[i][1], max_src_len, max_tgt_len)
        need_src = len(src_ids) + 1  # + EOS separator
        need_tgt = len(tgt_ids) + 2  # + BOS + EOS

        if cur_src and (
            len(cur_src) + need_src > max_src_len or len(cur_tgt) + need_tgt > max_tgt_len
        ):
            flush()

        s0 = len(cur_src)
        cur_src.extend(src_ids)
        cur_src.append(eos_id)
        cur_src_bounds.append((s0, len(cur_src)))

        t0 = len(cur_tgt)
        cur_tgt.append(bos_id)
        cur_tgt.extend(tgt_ids)
        cur_tgt.append(eos_id)
        cur_tgt_bounds.append((t0, len(cur_tgt)))

    flush()

    B = len(rows_src)
    src = torch.full((B, max_src_len), pad_id, dtype=torch.long)
    tgt = torch.full((B, max_tgt_len), pad_id, dtype=torch.long)
    for b in range(B):
        src[b, : len(rows_src[b])] = torch.tensor(rows_src[b], dtype=torch.long)
        tgt[b, : len(rows_tgt[b])] = torch.tensor(rows_tgt[b], dtype=torch.long)

    src_pad_mask = src == pad_id
    loss_mask = tgt[:, 1:] != pad_id

    src_attn = torch.full((B, 1, max_src_len, max_src_len), float("-inf"))
    tgt_attn = torch.full((B, 1, max_tgt_len, max_tgt_len), float("-inf"))
    cross_attn = torch.full((B, 1, max_tgt_len, max_src_len), float("-inf"))

    for b in range(B):
        for (s0, s1), (t0, t1) in zip(src_bounds[b], tgt_bounds[b], strict=True):
            src_attn[b, 0, s0:s1, s0:s1] = 0.0
            cross_attn[b, 0, t0:t1, s0:s1] = 0.0
            block = torch.zeros(t1 - t0, t1 - t0)
            block = block.masked_fill(
                torch.triu(torch.ones_like(block), diagonal=1).bool(), float("-inf")
            )
            tgt_attn[b, 0, t0:t1, t0:t1] = block

    # Padding rows have no block coverage -> all -inf scores -> NaN softmax.
    # Guarantee every query row sees at least one key: allow pad positions to
    # attend themselves (their outputs are excluded from the loss anyway).
    for b in range(B):
        for m in (src_attn[b, 0], tgt_attn[b, 0]):
            visible = m.max(dim=-1).values > float("-inf")
            dead_rows = torch.nonzero(~visible, as_tuple=False).flatten()
            if dead_rows.numel():
                m[dead_rows, dead_rows] = 0.0
        cm = cross_attn[b, 0]
        visible = cm.max(dim=-1).values > float("-inf")
        dead_rows = torch.nonzero(~visible, as_tuple=False).flatten()
        if dead_rows.numel():
            cm[dead_rows, :] = 0.0

    return {
        "src": src,
        "src_pad_mask": src_pad_mask,
        "tgt": tgt,
        "loss_mask": loss_mask,
        "src_attn_mask": src_attn,
        "tgt_attn_mask": tgt_attn,
        "cross_attn_mask": cross_attn,
        "n_packed": torch.tensor(packed_counts, dtype=torch.long),
    }


__all__ = ["pack_batch"]
