"""
cadgenesis.evaluation.hf_eval
=============================
Standardized evaluation helpers for CADGenesis-LM.

* ``cad_perplexity`` — mean per-token cross-entropy perplexity over CAD id
  sequences, scored teacher-forced through the model's forward contract
  ``model(src_ids, tgt_in_ids, tgt_type_ids, ...) -> (logits, confidence)``.
* ``run_lm_eval`` — thin, honest integration point for the optional
  ``lm_eval`` package (EleutherAI); validates the dependency and returns a
  status dict, leaving harness wiring to the caller.
* ``benchmark_suite`` — convenience bundle around ``cad_perplexity``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cadgenesis.transformer.losses import MaskedCrossEntropyLoss

__all__ = [
    "benchmark_suite",
    "cad_perplexity",
    "run_lm_eval",
]


def _type_ids(ids: list[int], vocab) -> list[int]:
    """Map token ids to token-family type ids (0 fallback for unknown ids)."""
    return [vocab.type_id_of(tok_id) if tok_id in vocab else 0 for tok_id in ids]


def cad_perplexity(
    model: nn.Module,
    tokenizer,
    cad_sequences: list[list[int]],
    src_ids: list[int] | None = None,
    batch_size: int = 8,
) -> float:
    """Mean per-token cross-entropy perplexity of CAD id sequences.

    Each sequence is scored teacher-forced: the decoder input is
    ``BOS + seq`` and the targets are ``seq + EOS``, padded with
    ``tokenizer.pad_id`` (ignored by the loss).  The model is an
    encoder-decoder, so a dummy source drives the encoder: ``src_ids``
    defaults to a single BOS token id — the ids must be valid for the
    model's language vocabulary (``[0, lang_vocab_size)``).

    The tokenizer contract used: ``tokenizer.bos_id``, ``tokenizer.eos_id``,
    ``tokenizer.pad_id`` and ``tokenizer.vocab.type_id_of(id)`` for type
    embeddings.  (``tokenizer.encode_cad_sequence`` exists for callers that
    hold token *strings*; this function consumes pre-tokenized id lists.)

    Returns ``exp(mean_ce)`` over all non-pad tokens.  Raises ``ValueError``
    on an empty input.
    """
    if not cad_sequences:
        raise ValueError("cad_sequences must be a non-empty list of id sequences")
    if any(not seq for seq in cad_sequences):
        raise ValueError("cad_sequences must not contain empty sequences")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if src_ids is None:
        bos_id = getattr(tokenizer, "bos_id", None)
        if bos_id is None:
            raise ValueError("src_ids is required when the tokenizer has no bos_id attribute")
        src_ids = [bos_id]

    device = next(model.parameters()).device
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    vocab = tokenizer.vocab

    loss_fn = MaskedCrossEntropyLoss(pad_id=pad_id)
    model.eval()

    ce_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for start in range(0, len(cad_sequences), batch_size):
            batch = cad_sequences[start : start + batch_size]
            full = [[bos_id, *seq, eos_id] for seq in batch]
            max_len = max(len(f) for f in full) - 1

            tgt_in: list[list[int]] = []
            tgt_type: list[list[int]] = []
            targets: list[list[int]] = []
            for f in full:
                t_in = f[:-1]
                tgt = f[1:]
                pad_n = max_len - len(t_in)
                tgt_in.append(t_in + [pad_id] * pad_n)
                tgt_type.append(_type_ids(t_in, vocab) + [0] * pad_n)
                targets.append(tgt + [pad_id] * pad_n)

            src = torch.tensor([src_ids], dtype=torch.long, device=device).expand(len(batch), -1)
            tgt_in_t = torch.tensor(tgt_in, dtype=torch.long, device=device)
            tgt_type_t = torch.tensor(tgt_type, dtype=torch.long, device=device)
            targets_t = torch.tensor(targets, dtype=torch.long, device=device)

            logits, _confidence = model(src, tgt_in_t, tgt_type_t)
            ce = loss_fn(logits, targets_t)
            n = int((targets_t != pad_id).sum().item())
            ce_sum += float(ce.item()) * n
            token_count += n

    if token_count == 0:
        raise ValueError("no non-pad tokens to score")
    mean_ce = ce_sum / token_count
    return float(math.exp(mean_ce))


def run_lm_eval(
    model: nn.Module,
    tokenizer,
    tasks: list[str] | None = None,
) -> dict:
    """Validate the optional ``lm_eval`` dependency and return a status dict.

    ``lm_eval`` (EleutherAI) is not a hard dependency of CADGenesis-LM.  When
    it is missing, an ``ImportError`` with install guidance is raised.  When
    present, this function returns
    ``{"tasks": [...], "status": "requires lm_eval integration adapter"}``:
    wiring a full harness (``loglikelihood``/``greedy_until`` adapters over
    the CAD tokenizer) is intentionally left to the caller — the
    ``cad_perplexity`` scoring path above documents the model's scoring
    contract to build such an adapter against.
    """
    try:
        import lm_eval  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "run_lm_eval requires the 'lm_eval' package (EleutherAI LM harness). "
            "Install it with: pip install lm-eval. This function cannot run "
            "without it."
        ) from exc

    chosen = ["mmlu"] if tasks is None else list(tasks)
    return {
        "tasks": chosen,
        "status": "requires lm_eval integration adapter",
    }


def benchmark_suite(
    model: nn.Module,
    tokenizer,
    cad_sequences: list[list[int]],
) -> dict:
    """Run the standard evaluation bundle over CAD sequences."""
    perplexity = cad_perplexity(model, tokenizer, cad_sequences)
    return {
        "perplexity": float(perplexity),
        "n_sequences": len(cad_sequences),
    }
