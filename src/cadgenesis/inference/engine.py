"""
cadgenesis.inference
===================
Production inference engine for CADGenesis-LM v2.0.

Decodes a natural-language design request into a CAD token sequence using the
Autonomous CAD Tokenizer and the Geometry-Aware / Self-Designing transformer.
It is model-agnostic: any object matching the forward contract
``model(src_ids, tgt_in_ids, tgt_type_ids, src_key_padding_mask=...) ->
(logits, confidence)`` works, including ``GeometryAwareTransformer`` and
``SelfDesigningTransformer``.

Features
--------
* Greedy decoding and beam search.
* Confidence scoring from the model's confidence head (sigmoid, per-token).
* TOON output — every result can be serialized via the tokenizer's TOON
  backend (``result.toon``).
* Optional self-designing telemetry when decoding with a
  ``SelfDesigningTransformer``.

Complexity
----------
    greedy():  O(T · fwd)   where T = generated tokens, fwd = one forward pass
    beam():    O(T · B · fwd)  with B = beam width
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from cadgenesis.tokenizer import AutonomousCADTokenizer


@dataclass
class GenerationResult:
    """Output of one decoding run."""

    text: str
    tokens: list[str] = field(default_factory=list)  # generated CAD tokens
    ids: list[int] = field(default_factory=list)  # generated CAD ids
    confidence: float = 0.0  # mean sigmoid confidence
    per_token_confidence: list[float] = field(default_factory=list)
    toon: str = ""  # TOON-serialized result
    stopped_on_eos: bool = False

    def parse(self):
        """Best-effort structured geometry parse (legacy validator)."""
        try:
            from examples.mini_demo.generate import parse_cad_sequence
        except ImportError as err:
            raise RuntimeError(
                "parse() requires the legacy examples/mini_demo/generate.py parser."
            ) from err
        return parse_cad_sequence(self.tokens)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(tokens={self.tokens}, "
            f"confidence={self.confidence:.3f}, eos={self.stopped_on_eos})"
        )


class CADInferenceEngine:
    """
    Greedy / beam autoregressive decoder for CAD sequence generation.

    Parameters
    ----------
    model : nn.Module
        GeometryAwareTransformer or SelfDesigningTransformer (or duck-typed
        equivalent returning ``(logits, confidence)``).
    tokenizer : AutonomousCADTokenizer
        Native tokenizer used for prompt encoding and output decoding.
    device : str, optional
        Compute device.  Defaults to CUDA if available else CPU.
    """

    def __init__(
        self,
        model,
        tokenizer: AutonomousCADTokenizer,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            # Infer the device from the model's own parameter device so a
            # CPU model is served on CPU even when CUDA is available (keeps
            # CPU-built engines and test environments consistent).
            first = next(model.parameters(), None)
            if first is not None and first.device.type == "cuda" and torch.cuda.is_available():
                device = f"cuda:{first.device.index}"
            else:
                device = "cpu"
        self.device = device
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model.to(device)

    # ----------------------------------------------------------- primitives

    def _encode_prompt(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.encode_text(text)
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        src_pad = src == self.tokenizer.pad_id
        return src, src_pad

    def _type_ids(self, tgt: torch.Tensor) -> torch.Tensor:
        types = []
        for tok_id in tgt[0].tolist():
            if tok_id in self.tokenizer.vocab:
                types.append(self.tokenizer.vocab.type_id_of(int(tok_id)))
            else:
                types.append(0)
        return torch.tensor([types], dtype=torch.long, device=self.device)

    def _logits_confidence(self, src, tgt, src_pad):
        tgt_type = self._type_ids(tgt)
        out = self.model(
            src,
            tgt,
            tgt_type,
            src_key_padding_mask=src_pad,
        )
        # Support both the (logits, confidence) tuple and bare logits.
        if isinstance(out, tuple):
            logits, confidence = out
        else:
            logits, confidence = out, None
        return logits, confidence

    def _mask_bos(self, row: torch.Tensor) -> torch.Tensor:
        """
        Forbid the BOS token as a *generated* token: BOS is the internal
        sequence-start marker (see ``_build_result``), never a legal CAD
        program token.  Masking it during candidate selection prevents
        degenerate infinite BOS loops from any (e.g. untrained) model.
        ``row`` is the (V,) log-probability/logit vector being sampled.
        """
        bos = self.tokenizer.bos_id
        if 0 <= bos < row.shape[-1]:
            row = row.clone()
            row[..., bos] = float("-inf")
        return row

    def _build_result(self, text, tgt, confs, stopped_eos) -> GenerationResult:
        ids = tgt[0].tolist()
        # Drop the leading BOS token from the reported sequence.
        content_ids = ids[1:] if ids and ids[0] == self.tokenizer.bos_id else ids
        tokens = self.tokenizer.decode_cad_sequence(content_ids)
        mean_conf = float(torch.sigmoid(torch.tensor(confs)).mean()) if confs else 0.0

        # Faithful TOON output: serialize the *actual* generated ids so
        # unknown (unregistered) ids survive the round-trip.
        seq = self.tokenizer.encode_cad_sequence(tokens, add_bos=False, add_eos=False)
        seq.cad_ids = content_ids
        seq.type_ids = [
            self.tokenizer.vocab.type_id_of(int(i)) if int(i) in self.tokenizer.vocab else 0
            for i in content_ids
        ]
        seq.attention_mask = [1] * len(content_ids)

        return GenerationResult(
            text=text,
            tokens=tokens,
            ids=content_ids,
            confidence=mean_conf,
            per_token_confidence=[float(c) for c in confs],
            toon=self.tokenizer.serialize_to_toon(seq),
            stopped_on_eos=stopped_eos,
        )

    # --------------------------------------------------------------- greedy

    @torch.no_grad()
    def greedy(self, text: str, max_len: int = 64, use_cache: bool = True) -> GenerationResult:
        """
        Greedy autoregressive decoding (argmax per step).

        ``use_cache`` enables incremental KV-cache decoding when the model
        implements ``prepare_decoder_cache``/``decode_step`` (GeometryAware
        Transformer); other models always fall back to full-sequence forwards.
        """
        self.model.eval()
        if (
            use_cache
            and hasattr(self.model, "prepare_decoder_cache")
            and hasattr(self.model, "decode_step")
        ):
            return self._greedy_cached(text, max_len)
        return self._greedy_full(text, max_len)

    @torch.no_grad()
    def _greedy_full(self, text: str, max_len: int) -> GenerationResult:
        src, src_pad = self._encode_prompt(text)
        tgt = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long, device=self.device)
        confs: list[float] = []
        stopped = False
        for _ in range(max_len):
            logits, confidence = self._logits_confidence(src, tgt, src_pad)
            if confidence is not None:
                confs.append(float(confidence[0, -1].item()))
            next_id = int(self._mask_bos(logits[0, -1]).argmax(-1).item())
            tgt = torch.cat([tgt, torch.tensor([[next_id]], device=self.device)], dim=1)
            if next_id == self.tokenizer.eos_id:
                stopped = True
                break
        return self._build_result(text, tgt, confs, stopped)

    @torch.no_grad()
    def _greedy_cached(self, text: str, max_len: int) -> GenerationResult:
        """
        KV-cache greedy decoding: encodes the prompt once, precomputes the
        encoder-side K/V (geometry + memory heads) and grows only the
        decoder self-attention K/V per step.
        """
        src, src_pad = self._encode_prompt(text)
        cache = self.model.prepare_decoder_cache(src, src_key_padding_mask=src_pad)
        tgt = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long, device=self.device)
        confs: list[float] = []
        stopped = False
        for _ in range(max_len):
            tgt_type = self._type_ids(tgt[:, -1:])
            logits, confidence = self.model.decode_step(tgt[:, -1:], tgt_type, cache)
            if confidence is not None:
                confs.append(float(confidence[0, -1].item()))
            next_id = int(self._mask_bos(logits[0, -1]).argmax(-1).item())
            tgt = torch.cat([tgt, torch.tensor([[next_id]], device=self.device)], dim=1)
            if next_id == self.tokenizer.eos_id:
                stopped = True
                break
        return self._build_result(text, tgt, confs, stopped)

    # ----------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample(
        self,
        text: str,
        max_len: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
        start_ids: list[int] | None = None,
    ) -> GenerationResult:
        """
        Temperature (optionally top-k) sampling decoding.

        ``start_ids`` seeds the sequence with already-generated ids (used by
        test-time search to continue from a prefix).  Backbone for RLVR
        rollouts and test-time compute (best-of-n, self-consistency, MCTS).
        ``temperature <= 0`` falls back to greedy.
        """
        if temperature <= 0:
            return self.greedy(text, max_len=max_len, use_cache=use_cache)
        self.model.eval()
        src, src_pad = self._encode_prompt(text)
        cached = (
            use_cache
            and hasattr(self.model, "prepare_decoder_cache")
            and hasattr(self.model, "decode_step")
        )
        prefix = start_ids if start_ids is not None else [self.tokenizer.bos_id]
        if not prefix:
            prefix = [self.tokenizer.bos_id]

        cache = None
        if cached:
            cache = self.model.prepare_decoder_cache(src, src_key_padding_mask=src_pad)
            for tok in prefix:
                step_ids = torch.tensor([[tok]], dtype=torch.long, device=self.device)
                step_type = self._type_ids(step_ids)
                self.model.decode_step(step_ids, step_type, cache)

        tgt = list(prefix)
        confs: list[float] = []
        stopped = False
        budget = max_len - len(tgt)
        for _ in range(budget):
            if cached:
                step_ids = torch.tensor([[tgt[-1]]], dtype=torch.long, device=self.device)
                step_type = self._type_ids(step_ids)
                logits, confidence = self.model.decode_step(step_ids, step_type, cache)
            else:
                tgt_t = torch.tensor([tgt], dtype=torch.long, device=self.device)
                logits, confidence = self._logits_confidence(src, tgt_t, src_pad)
                logits = logits[:, -1]

            if confidence is not None:
                confs.append(float(confidence[0, -1].item()))

            last = logits[0, -1]
            if temperature != 1.0:
                last = last / temperature
            if top_k:
                k = min(top_k, last.shape[-1])
                topv, _ = torch.topk(last, k)
                last = torch.where(last >= topv[-1], last, torch.full_like(last, float("-inf")))
            last = self._mask_bos(last)
            probs = torch.softmax(last, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            tgt.append(next_id)
            if next_id == self.tokenizer.eos_id:
                stopped = True
                break

        tgt_t = torch.tensor([tgt], dtype=torch.long, device=self.device)
        return self._build_result(text, tgt_t, confs, stopped)

    # ----------------------------------------------------------------- beam

    @staticmethod
    def _length_penalty(length: int, alpha: float) -> float:
        """
        GNMT-style length normalization: ``score / ((5 + len) / 6) ** alpha``.
        ``alpha=0`` disables the penalty (plain cumulative log-probability).
        """
        return ((5.0 + length) / 6.0) ** alpha

    @torch.no_grad()
    def beam(
        self,
        text: str,
        beam_width: int = 3,
        max_len: int = 64,
        length_penalty: float = 0.6,
    ) -> GenerationResult:
        """
        Beam search decoding (v6.1 §4.8):

        * **EOS handling** — beams that emit EOS are retired into a finished
          set and never expanded again; the search stops when ``beam_width``
          hypotheses are finished or ``max_len`` tokens are generated.
        * **Length normalization** — hypotheses are compared by
          ``score / ((5 + len) / 6) ** length_penalty`` (GNMT) so a shorter,
          higher-quality program beats a verbose one; ``length_penalty=0``
          restores plain cumulative log-probability.
        * **Score normalization** — the final pick is the best *normalized*
          hypothesis across both the finished set and the best unfinished
          beam (instead of always preferring a finished one).
        * **Finite-score filtering** — candidates carrying -inf/NaN scores
          (e.g. tokens drawn from a `topk` over a masked distribution) can
          never win and are dropped before the beam set is pruned, so the
          search never expands garbage hypotheses.
        """
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1.")
        self.model.eval()
        src, src_pad = self._encode_prompt(text)

        beams: list[tuple[list[int], float, list[float]]] = [([self.tokenizer.bos_id], 0.0, [])]
        finished: list[tuple[list[int], float, list[float]]] = []
        eos = self.tokenizer.eos_id

        for _ in range(max_len):
            if len(finished) >= beam_width:
                break
            candidates: list[tuple[list[int], float, list[float]]] = []
            for ids, score, confs in beams:
                tgt = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits, confidence = self._logits_confidence(src, tgt, src_pad)
                probs = self._mask_bos(torch.log_softmax(logits[0, -1], dim=-1))
                top_probs, top_idx = probs.topk(beam_width)
                new_confs = confs + (
                    [float(confidence[0, -1].item())] if confidence is not None else [0.0]
                )
                for p, idx in zip(top_probs.tolist(), top_idx.tolist(), strict=False):
                    cand_score = score + p
                    if not math.isfinite(cand_score):
                        continue
                    cand_ids = [*ids, int(idx)]
                    cand = (cand_ids, cand_score, new_confs)
                    if int(idx) == eos:
                        # Retire EOS beams: they are complete and must never
                        # be expanded further.
                        finished.append(cand)
                    else:
                        candidates.append(cand)
            # Keep the best beam_width unfinished beams (dedupe prefixes).
            seen: set = set()
            kept: list[tuple[list[int], float, list[float]]] = []
            for ids, score, confs in sorted(candidates, key=lambda c: c[1], reverse=True):
                key = tuple(ids)
                if key in seen:
                    continue
                seen.add(key)
                kept.append((ids, score, confs))
                if len(kept) >= beam_width:
                    break
            beams = kept
            if not beams:
                break

        # Final selection: best normalized score over finished hypotheses and
        # the best unfinished beam.
        def norm(score: float, length: int) -> float:
            return score / self._length_penalty(length, length_penalty)

        final: tuple[list[int], float, list[float]] | None = None
        final_norm = float("-inf")
        for cand in [*finished, *beams]:
            ids, score, confs = cand
            n = norm(score, len(ids))
            if n > final_norm:
                final_norm = n
                final = cand

        assert final is not None
        ids, score, confs = final
        tgt = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self._build_result(text, tgt, confs, ids[-1] == eos)

    # --------------------------------------------------------------- speculative

    def _ngram_draft(
        self,
        ids: list[int],
        n: int,
        k: int,
    ) -> list[int]:
        """
        N-gram drafter: find the most recent occurrence of the last ``n`` ids
        in the history (skipping the tail itself) and return the ``k`` tokens
        that followed it.  Returns [] when no repetition is found — the caller
        then falls back to a single-step greedy advance.
        """
        if len(ids) < n + 1:
            return []
        needle = tuple(ids[-n:])
        best: tuple[list, int] | None = None  # (following tokens, end index)
        for start in range(len(ids) - n):
            if tuple(ids[start : start + n]) == needle:
                end = start + n
                if end >= len(ids) - 1:
                    continue
                follow = ids[end : end + k]
                # Prefer the longest follow-up / earliest occurrence.
                if best is None or len(follow) > len(best[0]) or start < best[1]:
                    best = (follow, start)
        return best[0] if best else []

    @torch.no_grad()
    def speculative(
        self,
        text: str,
        max_len: int = 64,
        num_speculative_tokens: int = 4,
        ngram: int = 2,
    ) -> GenerationResult:
        """
        N-gram speculative decoding.

        At each round the engine drafts ``num_speculative_tokens`` candidates
        with an n-gram repetition model, then verifies them against the target
        model in a single causal forward.  Accepted tokens are committed; on
        the first mismatch the target's argmax token is taken (greedy-preserving),
        so the output is always exactly what greedy decoding would produce.
        """
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1.")
        self.model.eval()
        src, src_pad = self._encode_prompt(text)
        tgt = [self.tokenizer.bos_id]
        confs: list[float] = []
        stopped = False

        while len(tgt) - 1 < max_len:
            remaining = max_len - (len(tgt) - 1)
            # Reserve one slot for the greedy tail advance so a round never
            # overshoots max_len (drafts + tail == remaining, exactly).
            k = min(num_speculative_tokens, remaining - 1)
            draft = self._ngram_draft(tgt, ngram, k)
            candidate = tgt + (draft or [])

            # One causal forward over BOS + candidate (prompt on the encoder side).
            tgt_t = torch.tensor([candidate], dtype=torch.long, device=self.device)
            tgt_type = self._type_ids(tgt_t)
            out = self.model(
                src,
                tgt_t,
                tgt_type,
                src_key_padding_mask=src_pad,
            )
            logits, confidence = out if isinstance(out, tuple) else (out, None)
            if confidence is not None:
                confs.append(float(confidence[0, -1].item()))

            base = len(tgt) - 1
            for pos, draft_id in enumerate(draft):
                pred = int(self._mask_bos(logits[0, base + pos]).argmax(-1).item())
                if pred == draft_id:
                    tgt.append(draft_id)
                else:
                    # Greedy-preserving fallback at the first mismatch.
                    tgt.append(pred)
                    break
            else:
                # No draft to verify (empty) or every draft token accepted:
                # advance one real step from the sequence tail.
                next_id = int(self._mask_bos(logits[0, -1]).argmax(-1).item())
                tgt.append(next_id)

            if tgt[-1] == self.tokenizer.eos_id:
                stopped = True
                break

        tgt_t = torch.tensor([tgt], dtype=torch.long, device=self.device)
        return self._build_result(text, tgt_t, confs, stopped)

    # --------------------------------------------------------------- batch

    @torch.no_grad()
    def batch_generate(
        self,
        texts: list[str],
        max_len: int = 64,
        beam_width: int | None = None,
    ) -> list[GenerationResult]:
        """Run greedy (or beam, if ``beam_width`` given) over several prompts."""
        if beam_width:
            return [self.beam(t, beam_width=beam_width, max_len=max_len) for t in texts]
        return [self.greedy(t, max_len=max_len) for t in texts]

    # ------------------------------------------------------------- telemetry

    def self_design_report(self) -> dict | None:
        """Architecture / routing telemetry when using a SelfDesigningTransformer."""
        report = getattr(self.model, "architecture_report", None)
        if report is None:
            return None
        return report()
