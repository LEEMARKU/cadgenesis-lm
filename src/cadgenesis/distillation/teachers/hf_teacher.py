"""
cadgenesis.distillation.teachers.hf_teacher
===========================================
Generic HuggingFace teacher adapter for LLM-to-LLM teaching.

Any instruction-tuned causal LM on the Hub (or a local directory) can be used
as the *teacher*: ``Qwen/Qwen2.5-1.5B-Instruct``, ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B``,
your own fine-tuned model, etc.  The teacher maps natural-language part
descriptions to CAD feature token lists which become training data for the
student model.

The adapter is deliberately model-agnostic:

* chat-template based prompting (falls back to plain text when a template is
  missing),
* bf16 on CUDA (or fp32 CPU),
* a DSL-aware parser that extracts only known CAD feature tokens from the
  teacher's free-form output, and
* the same few-shot system prompt used across all teacher models.

Usage::

    from cadgenesis.distillation.teachers.hf_teacher import HFTeacher

    teacher = HFTeacher("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    tokens = teacher.generate_cad_program("a steel box")   # ["SKETCH_RECT", "EXTRUDE", "BOX"]
"""

from __future__ import annotations

import re
from typing import Any

import torch

CAD_DEFAULT_VOCAB: tuple[str, ...] = (
    "SKETCH_RECT",
    "SKETCH_CIRCLE",
    "SKETCH_LINE",
    "SKETCH_ARC",
    "EXTRUDE",
    "REVOLVE",
    "SWEEP",
    "LOFT",
    "BOX",
    "CYLINDER",
    "SPHERE",
    "CONE",
    "WEDGE",
    "TORUS",
    "HOLE",
    "CHAMFER",
    "FILLET",
    "SHELL",
    "RIB",
    "DRAFT",
)

# System prompt shared by every HF teacher: concise, few-shot, no reasoning.
DEFAULT_SYSTEM_PROMPT = (
    "You convert natural-language CAD part descriptions into comma-separated "
    "CAD feature lists. Reply with ONLY the list of feature tokens, nothing "
    "else. No reasoning, no prose.\n\n"
    "Examples:\n"
    "User: a steel box\n"
    "Assistant: SKETCH_RECT, EXTRUDE, BOX\n\n"
    "User: a cylindrical housing with 50mm radius and 55mm height\n"
    "Assistant: CYLINDER, EXTRUDE, BOX\n\n"
    "User: a mounting bracket holding a 20mm rod\n"
    "Assistant: SKETCH_RECT, EXTRUDE, BOX, CYLINDER\n\n"
    "User: a 115mm base plate with 135mm spherical feet\n"
    "Assistant: SKETCH_RECT, EXTRUDE, SPHERE"
)


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "HFTeacher requires `transformers`; install with "
            "`pip install transformers huggingface_hub safetensors accelerate`."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


class HFTeacher:
    """Generic teacher adapter over any HuggingFace causal language model.

    Parameters
    ----------
    model_id:
        HuggingFace model id or local directory (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``).
    device:
        ``"cuda"`` or ``"cpu"``.
    torch_dtype:
        Weights dtype (default bfloat16 on CUDA, float32 on CPU).
    max_new_tokens:
        Generation budget per prompt (default 64).
    temperature:
        Sampling temperature (0 = greedy).
    system_prompt:
        Few-shot system prompt; defaults to :data:`DEFAULT_SYSTEM_PROMPT`.
    allowed_tokens:
        DSL tokens the parser accepts.  Defaults to :data:`CAD_DEFAULT_VOCAB`.
        Pass ``None`` to accept any ``UPPER_SNAKE`` token.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        allowed_tokens: tuple[str, ...] | None = CAD_DEFAULT_VOCAB,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if device.startswith("cuda") else torch.float32
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.allowed_tokens = set(allowed_tokens) if allowed_tokens else None
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------- lifecycle

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        """Load tokenizer + weights (idempotent)."""
        if self.loaded:
            return
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        tok = AutoTokenizer.from_pretrained(self.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        model.eval()
        self._tokenizer = tok
        self._model = model

    @property
    def tokenizer(self) -> Any:
        self.load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        self.load()
        return self._model

    # -------------------------------------------------------------- prompting

    def _build_prompt(self, prompt: str) -> str:
        """Wrap ``prompt`` using the model's chat template (or plain text)."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Raw teacher completion for a single prompt."""
        text = self._build_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9
        out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ---------------------------------------------------------------- parsing

    def parse_cad_tokens(self, text: str) -> list[str]:
        """Extract known CAD feature tokens from teacher output.

        Unknown tokens are skipped, so a rambling R1-style response still
        yields a clean program whenever the feature list appears anywhere.
        """
        found = re.findall(r"\b[A-Z][A-Z0-9_]{1,31}\b", text or "")
        if self.allowed_tokens is not None:
            return [t for t in found if t in self.allowed_tokens]
        return found

    def generate_cad_program(self, prompt: str) -> list[str]:
        """Return the parsed CAD feature-token program for a prompt."""
        return self.parse_cad_tokens(self.generate(prompt))

    def generate_feature_record(self, prompt: str) -> dict[str, Any]:
        """One ``{"text": prompt, "cad": [tokens]}`` record (train.py shape)."""
        return {"text": prompt, "cad": self.generate_cad_program(prompt)}


__all__ = [
    "CAD_DEFAULT_VOCAB",
    "DEFAULT_SYSTEM_PROMPT",
    "HFTeacher",
]
