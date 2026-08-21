"""
cadgenesis.inference.eagle
==========================
EAGLE-style speculative decoding with a learned draft head.

Modern speculative decoding (EAGLE-1/2/3, Medusa) replaces heuristic drafters
(n-grams, lookahead) with a *small model that predicts the next token from the
target model's hidden state*.  Because the target model already computes a
hidden state per token, the draft head only needs a cheap transformation to
predict the next token — and the target then *verifies* the draft in one
forward, committing all accepted tokens.

This module provides:

* :class:`EagleDraftHead` — a lightweight causal transformer mapping the
  target's per-token hidden states to next-token logits.
* :func:`train_eagle` — collect (hidden, next_token) pairs from the target
  model and fine-tune the head with cross-entropy.
* :func:`speculative_eagle` — engine integration.  Greedy-preserving: the
  output is always exactly what greedy decoding would produce, because every
  drafted token is verified and mismatches fall back to the target's argmax.

The draft head is *decoupled* from the target model, so it can be trained on
real data later without touching the main checkpoint.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from cadgenesis.inference.engine import CADInferenceEngine, GenerationResult
from cadgenesis.transformer.attention import SelfAttention


class EagleDraftHead(nn.Module):
    """
    Small causal transformer: hidden state -> next-token logits (EAGLE-style).

    Parameters
    ----------
    d_model : int
        Target model width (also the draft width).
    num_heads : int
        Draft attention heads.
    ffn_dim : int, optional
        Draft FFN width (default ``2 * d_model``).
    vocab_size : int
        Target vocabulary size (draft logits space).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        vocab_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1.")
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim or (2 * d_model)),
            nn.GELU(),
            nn.Linear(ffn_dim or (2 * d_model), d_model),
            nn.Dropout(dropout),
        )
        self.next_hidden = nn.Linear(d_model, d_model)
        self.logits_proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, d_model) -> draft logits (B, T, vocab_size).
        ``logits[:, t]`` predicts the token *after* position ``t``.
        """
        return self.predict_next(hidden)[0]

    def predict_next(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden: (B, T, d_model) -> (draft_logits (B, T, V), next_hidden (B, T, d)).
        ``next_hidden[:, t]`` is the draft-side prediction of the hidden state at
        ``t + 1`` — used to extend the draft autoregressively without running
        the target model (EAGLE's key trick).
        """
        x = hidden
        x = x + self.attn(self.norm1(x), attn_mask=None)
        x = x + self.ffn(self.norm2(x))
        next_hidden = self.next_hidden(x)
        return self.logits_proj(next_hidden), next_hidden

    @torch.no_grad()
    def draft(self, hidden: torch.Tensor) -> torch.Tensor:
        """Last-position draft logits (B, vocab_size)."""
        return self.forward(hidden)[:, -1]

    @torch.no_grad()
    def draft_sequence(self, hidden: torch.Tensor, k: int) -> list[int]:
        """
        Autoregressively draft ``k`` next-token candidates from the last hidden
        state, using the draft model's own predicted hidden states (EAGLE).
        """
        if k < 1:
            return []
        h = hidden
        ds: list[int] = []
        for _ in range(k):
            logits, nh = self.predict_next(h)
            d = int(logits[:, -1].argmax(-1).item())
            ds.append(d)
            h = nh[:, -1:]
        return ds


def collect_hidden_pairs(
    model: nn.Module,
    sequences: list[list[int]],
    type_ids_fn,
    max_len: int = 64,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Run the target model over each sequence and collect
    ``(hidden_state, next_token)`` pairs for training the draft head.

    ``type_ids_fn(token_id) -> type_id`` maps ids to the feature-type ids the
    model expects.  Returns a list of ``(hidden (1, T-1, d), targets (1, T-1))``.
    """
    model.eval()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for seq in sequences:
        tgt = torch.tensor([seq])
        types = torch.tensor([[type_ids_fn(int(i)) for i in seq]])
        with torch.no_grad():
            model(tgt, tgt, types)  # decoder-side forward; encoder unused
        hidden = cast(torch.Tensor | None, model.decode_hidden_states)
        if hidden is None or hidden.shape[1] < 2:
            continue
        pairs.append((hidden[:, :-1], torch.tensor([seq[1:]])[:, : hidden.shape[1] - 1]))
    return pairs


def train_eagle(
    head: EagleDraftHead,
    model: nn.Module,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    steps: int = 60,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """
    Train ``head`` on (hidden, next_token) pairs with cross-entropy.
    Returns the per-step loss history.
    """
    if not pairs:
        raise ValueError("train_eagle requires at least one (hidden, target) pair.")
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    losses: list[float] = []
    all_hidden = [h.to(device) for h, _ in pairs]
    all_targets = [t.to(device) for _, t in pairs]
    for _ in range(steps):
        optimizer.zero_grad()
        total = torch.zeros((), device=device)
        for hidden, target in zip(all_hidden, all_targets, strict=True):
            logits = head(hidden)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
            )
            total = total + loss
        total.backward()
        optimizer.step()
        losses.append(total.item())
    return losses


def speculative_eagle(
    engine: CADInferenceEngine,
    text: str,
    draft_head: EagleDraftHead,
    max_len: int = 64,
    num_speculative_tokens: int = 4,
    device: str | None = None,
) -> GenerationResult:
    """
    EAGLE-style speculative decoding, greedy-preserving.

    Each round: read the target model's last-token hidden state, ask the draft
    head for the next token, verify it with the target's ``decode_step``, and
    commit every accepted token.  On the first mismatch the target's argmax is
    taken instead — so the result always equals plain greedy decoding.
    """
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1.")
    device = device or engine.device
    model = engine.model
    model.eval()
    draft_head.eval()

    src, src_pad = engine._encode_prompt(text)
    cache = model.prepare_decoder_cache(src, src_key_padding_mask=src_pad)
    tgt = [engine.tokenizer.bos_id]
    confs: list[float] = []
    stopped = False

    while len(tgt) - 1 < max_len:
        remaining = max_len - (len(tgt) - 1)
        limit = min(num_speculative_tokens, remaining - 1)
        # Process the last appended token (position n-1); argmax predicts the
        # token at position n — greedy's ground truth.  This also refreshes
        # ``decode_hidden_states`` for the draft.
        step_ids = torch.tensor([[tgt[-1]]], dtype=torch.long, device=device)
        step_type = engine._type_ids(step_ids)
        logits, confidence = model.decode_step(step_ids, step_type, cache)
        if confidence is not None:
            confs.append(float(confidence[0, -1].item()))
        greedy_id = int(engine._mask_bos(logits[0, -1]).argmax(-1).item())

        hidden = model.decode_hidden_states
        draft = draft_head.draft_sequence(hidden, limit)
        if draft and draft[0] == greedy_id:
            tgt.append(draft[0])
            if tgt[-1] == engine.tokenizer.eos_id:
                stopped = True
                break
            # Verify drafts i=1..: processing draft[i-1] predicts position
            # n+i, which must equal draft[i] to be accepted.
            for i in range(1, limit):
                step_ids = torch.tensor([[tgt[-1]]], dtype=torch.long, device=device)
                step_type = engine._type_ids(step_ids)
                logits, confidence = model.decode_step(step_ids, step_type, cache)
                if confidence is not None:
                    confs.append(float(confidence[0, -1].item()))
                g = int(engine._mask_bos(logits[0, -1]).argmax(-1).item())
                if draft[i] == g:
                    tgt.append(draft[i])
                else:
                    tgt.append(g)
                    if g == engine.tokenizer.eos_id:
                        stopped = True
                    break
                if tgt[-1] == engine.tokenizer.eos_id:
                    stopped = True
                    break
            else:
                # Every draft accepted: one real greedy tail step.
                step_ids = torch.tensor([[tgt[-1]]], dtype=torch.long, device=device)
                step_type = engine._type_ids(step_ids)
                logits, confidence = model.decode_step(step_ids, step_type, cache)
                if confidence is not None:
                    confs.append(float(confidence[0, -1].item()))
                tgt.append(int(engine._mask_bos(logits[0, -1]).argmax(-1).item()))
                if tgt[-1] == engine.tokenizer.eos_id:
                    stopped = True
            if stopped:
                break
        else:
            # Draft rejected (or empty): take the greedy choice.
            tgt.append(greedy_id)
            if greedy_id == engine.tokenizer.eos_id:
                stopped = True
                break

    tgt_t = torch.tensor([tgt], dtype=torch.long, device=device)
    return engine._build_result(text, tgt_t, confs, stopped)
