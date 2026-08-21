"""
cadgenesis.adapters.deepseek_r1
===============================
Use the open-weights reasoning model ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B``
as the *teacher* for training the CADGenesis-LM.

Why this is the right integration
---------------------------------
The CAD model's tokenizer/vocab (CAD feature tokens) is incompatible with
Qwen's ~152k-token vocab, so logits-level distillation is not possible.  The
standard, robust pattern is *synthetic-data distillation*: the teacher
generates R1-style reasoning traces and structured CAD programs for a prompt;
a quality filter (the CAD execution engine) keeps only verifiable programs;
and those ``(prompt, program)`` pairs become SFT / RLVR training data for the
CAD model.  This module provides exactly that loop with a *local* teacher (no
API keys, no data leaving the box).

Components
----------
* :class:`DeepSeekR1Reasoner` — lazy-loads the HF model (CPU-friendly) and
  generates reasoning traces / chat completions / hidden states.
* :class:`DeepSeekR1Teacher` — teacher surface producing structured CAD specs
  plus the parsed feature-token program (matching :class:`TeacherModelInterface`).
* :class:`DeepSeekR1DataGenerator` — the closed loop: prompts → teacher →
  quality filter → ``(prompt, cad_program_ids)`` pairs ready for training.

The HF model is loaded lazily (and can be injected in tests), so importing
this module never requires ``transformers`` to be present.
"""

from __future__ import annotations

import re
from typing import Any, cast

import torch
import torch.nn as nn

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Feature names we look for inside the teacher's free-form output.  The CAD
# tokenizer registers these under its feature families; anything the teacher
# produces outside them is simply ignored / rejected by the quality filter.
_FEATURE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,31}\b")


def _import_transformers():
    """Import the HF stack lazily so the rest of the codebase stays light."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "DeepSeek-R1 integration requires `transformers`; install with "
            "`pip install transformers huggingface_hub safetensors accelerate`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


class DeepSeekR1Reasoner:
    """
    Local wrapper around ``DeepSeek-R1-Distill-Qwen-1.5B``.

    The model is loaded on first use.  ``model`` / ``tokenizer`` can be
    injected directly (useful in tests or when the caller already loaded them).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        cache_dir: str | None = None,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.cache_dir = cache_dir
        self._model: Any = model
        self._tokenizer = tokenizer

    # ------------------------------------------------------------- lifecycle

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        """Load the tokenizer and weights (idempotent)."""
        if self.loaded:
            return
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype_kwarg = self._dtype_kwarg()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **dtype_kwarg,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        self._model.to(self.device)
        self._model.eval()

    def _dtype_kwarg(self) -> dict[str, Any]:
        """transformers v5 renamed ``torch_dtype`` -> ``dtype``."""
        from transformers import __version__

        if __version__.startswith("4"):
            return {"torch_dtype": self.torch_dtype}
        return {"dtype": self.torch_dtype}

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    @property
    def model(self) -> nn.Module:
        self.load()
        assert self._model is not None
        return self._model

    # ------------------------------------------------------------ generation

    def _build_prompt(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Chat completion against the prompt (R1-style response text)."""
        text = self._build_prompt(prompt, system)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        sampling = temperature if temperature is not None else self.temperature
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": sampling > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if sampling > 0.0:
            gen_kwargs["temperature"] = sampling
            gen_kwargs["top_p"] = self.top_p
        out = cast(Any, self.model).generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_reasoning(self, prompt: str, **kwargs: Any) -> str:
        """R1-style reasoning trace for a design prompt (CoT distillation data)."""
        system = (
            "You are a mechanical design engineer. Think step by step about the "
            "design requirements, then propose a concrete parametric CAD program."
        )
        return self.generate(prompt, system=system, **kwargs)

    @torch.no_grad()
    def last_hidden_state(self, text: str) -> torch.Tensor:
        """Last-layer hidden state of the prompt (B, S, d_qwen) for cross-encoder use."""
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]


class DeepSeekR1Teacher:
    """
    Teacher surface that produces structured CAD specs from prompts.

    Compatible with :class:`~cadgenesis.distillation.distill_pipeline.TeacherModelInterface`
    (via :meth:`generate_cad_toon`) and adds :meth:`generate_cad_program`, which
    returns the parsed feature-token program ids directly usable for training.
    """

    def __init__(self, reasoner: DeepSeekR1Reasoner | None = None, **kwargs: Any):
        self.reasoner = reasoner or DeepSeekR1Reasoner(**kwargs)

    _SPEC_SYSTEM = (
        "Output ONLY a comma-separated list of CAD feature names, e.g.: "
        "SKETCH_RECT, EXTRUDE, BOX. No prose, no markdown."
    )

    def generate_cad_spec(self, prompt: str) -> str:
        """Ask the teacher for a structured CAD feature spec (raw text)."""
        return self.reasoner.generate(prompt, system=self._SPEC_SYSTEM, max_new_tokens=64)

    def generate_cad_toon(self, prompt: str) -> str:
        """TeacherModelInterface-compatible alias returning the raw spec."""
        return self.generate_cad_spec(prompt)

    def parse_feature_tokens(self, text: str, vocab) -> list[int]:
        """
        Map feature names in ``text`` onto the CAD tokenizer's ids (unknown
        names are skipped).  Returns the program as a list of token ids.
        """
        tok2id = vocab.to_tok2id() if hasattr(vocab, "to_tok2id") else vocab
        return [
            int(tok2id[token]) for token in _FEATURE_TOKEN_RE.findall(text or "") if token in tok2id
        ]

    def parse_feature_tokens_str(self, text: str) -> list[str]:
        """Feature-name *strings* in ``text`` (vocab-independent)."""
        return _FEATURE_TOKEN_RE.findall(text or "")

    def generate_cad_program(self, prompt: str, vocab) -> tuple[list[int], str]:
        """Return ``(program_ids, spec_text)`` for a prompt."""
        spec = self.generate_cad_spec(prompt)
        return self.parse_feature_tokens(spec, vocab), spec


class DeepSeekR1DataGenerator:
    """
    Closed loop: prompts → DeepSeek-R1 teacher → quality filter → training pairs.

    ``generate_dataset`` yields one record per prompt:

        {"prompt", "reasoning", "spec_text", "program_ids", "valid"}

    ``valid`` is True when at least one feature token was parsed AND the
    program passes the optional ``validator`` callable (default: a completion
    is valid if it is non-empty).  Pass a real oracle (e.g.
    :class:`~cadgenesis.distillation.rlvr.DesignOracle`) to filter on
    *verified* geometry.
    """

    def __init__(
        self,
        teacher: DeepSeekR1Teacher,
        vocab=None,
        validator=None,
        verbose: bool = False,
    ):
        self.teacher = teacher
        self.vocab = vocab
        self.validator = validator
        self.verbose = verbose

    def generate_record(self, prompt: str, reasoning: bool = True) -> dict[str, Any]:
        program_ids, spec = self.teacher.generate_cad_program(prompt, self.vocab)
        record: dict[str, Any] = {
            "prompt": prompt,
            "spec_text": spec,
            "program_ids": program_ids,
            "valid": bool(program_ids),
        }
        if reasoning:
            record["reasoning"] = self.teacher.reasoner.generate_reasoning(prompt)
        if self.validator is not None and program_ids:
            record["valid"] = bool(self.validator(program_ids))
        return record

    def generate_dataset(self, prompts: list[str], reasoning: bool = True) -> list[dict[str, Any]]:
        records = []
        for prompt in prompts:
            record = self.generate_record(prompt, reasoning=reasoning)
            records.append(record)
            if self.verbose:
                print(
                    f"[DeepSeekR1] {prompt!r} -> {len(record['program_ids'])} "
                    f"feature tokens, valid={record['valid']}"
                )
        return records

    def generate_feature_records(
        self, prompts: list[str], reasoning: bool = True
    ) -> list[dict[str, Any]]:
        """Teacher-generated records in the *standard train.py shape*.

        Returns ``{"text": prompt, "cad": [feature token strings]}`` plus an
        optional ``"reasoning"`` trace, so the output can be written to a JSONL
        dataset and consumed by :class:`~cadgenesis.datasets.cad_jsonl.CADJsonlDataset`
        (the tokenizer registers every feature name it uses automatically).
        """
        records: list[dict[str, Any]] = []
        for prompt in prompts:
            spec = self.teacher.generate_cad_spec(prompt)
            record: dict[str, Any] = {
                "text": prompt,
                "cad": self.teacher.parse_feature_tokens_str(spec),
            }
            if reasoning:
                record["reasoning"] = self.teacher.reasoner.generate_reasoning(prompt)
            records.append(record)
            if self.verbose:
                print(f"[DeepSeekR1] {prompt!r} -> {record['cad']}")
        return records


class MockDeepSeekR1Teacher:
    """Instant teacher stand-in for tests / CPU demos (no model download)."""

    PROGRAM = ["SKETCH_RECT", "EXTRUDE", "BOX"]

    def generate_cad_spec(self, prompt: str) -> str:
        return ", ".join(self.PROGRAM)

    def generate_cad_toon(self, prompt: str) -> str:
        return self.generate_cad_spec(prompt)

    def parse_feature_tokens(self, text: str, vocab) -> list[int]:
        return [
            int(vocab.to_tok2id()[token])
            for token in _FEATURE_TOKEN_RE.findall(text or "")
            if token in vocab.to_tok2id()
        ]

    def parse_feature_tokens_str(self, text: str) -> list[str]:
        return _FEATURE_TOKEN_RE.findall(text or "")

    def generate_cad_program(self, prompt: str, vocab) -> tuple[list[int], str]:
        spec = self.generate_cad_spec(prompt)
        return self.parse_feature_tokens(spec, vocab), spec

    def generate_reasoning(self, prompt: str) -> str:
        return f"[mock] {prompt} -> sketch, extrude, box"

    @property
    def reasoner(self):
        return self


__all__ = [
    "DEFAULT_MODEL_ID",
    "DeepSeekR1DataGenerator",
    "DeepSeekR1Reasoner",
    "DeepSeekR1Teacher",
    "MockDeepSeekR1Teacher",
]
