"""cadgenesis.distillation.teachers.open_source_teacher
======================================================
Open-source (DeepSeek/Qwen/Claude) teacher adapters.

One :class:`OpenSourceTeacher` adapter covers the three provider families:

* ``deepseek`` -- DeepSeek chat API (``deepseek-chat``), OpenAI-compatible.
* ``qwen`` -- Qwen via the DashScope OpenAI-compatible endpoint.
* ``claude`` -- Anthropic Claude (``claude-3-5-sonnet`` default).

Client libraries are imported lazily behind try/except (same pattern as
``openai_teacher.py`` / ``hf_teacher.py``): when the package is not
installed, no API key is configured, or the API call fails, the adapter
falls back to the rule-based generator of
:class:`~cadgenesis.distillation.distill_pipeline.TeacherModelInterface`
and never crashes.
"""

from __future__ import annotations

import os
from typing import Any

from cadgenesis.distillation.distill_pipeline import TeacherModelInterface
from sdk import toon_extended

__all__ = ["OpenSourceTeacher"]

#: Default model id per provider family.
MODEL_MAP: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "qwen": "qwen2.5-72b-instruct",
    "claude": "claude-3-5-sonnet-20241022",
}

#: OpenAI-compatible base URLs for the deepseek/qwen families.
_OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

#: Environment variables used to auto-discover API keys.
_API_KEY_ENV: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

_SYSTEM_PROMPT = (
    "You convert natural-language CAD part descriptions into a single TOON payload "
    "(header, optional schema and one or more data rows). Reply with only the TOON payload."
)


def _import_openai_client() -> Any | None:
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI


def _import_anthropic_client() -> Any | None:
    try:
        from anthropic import Anthropic
    except Exception:
        return None
    return Anthropic


class OpenSourceTeacher(TeacherModelInterface):
    """DeepSeek/Qwen/Claude teacher adapter with rule-based fallback.

    Parameters
    ----------
    provider:
        One of ``"deepseek"``, ``"qwen"``, ``"claude"``.
    model:
        Optional model id; defaults to :data:`MODEL_MAP` for the provider.
    api_key:
        Optional API key; falls back to the provider's ``*_API_KEY`` env
        var and to the rule-based generator when absent.
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if provider not in MODEL_MAP:
            raise ValueError(f"provider must be one of {sorted(MODEL_MAP)}, got {provider!r}")
        resolved_key = api_key or os.getenv(_API_KEY_ENV[provider])
        super().__init__(provider=provider, api_key=resolved_key)
        self.model = model or MODEL_MAP[provider]
        self._client: Any | None = None
        if self.api_key:
            if provider == "claude":
                client_cls = _import_anthropic_client()
            else:
                client_cls = _import_openai_client()
            if client_cls is not None:
                self._client = client_cls(
                    api_key=self.api_key, base_url=_OPENAI_COMPATIBLE_BASE_URLS[provider]
                )

    def generate_cad_toon(self, prompt: str) -> str:
        """Query the provider API for a TOON payload.

        Falls back to the rule-based generator of the base class when the
        client is unavailable, no API key is configured, the request
        raises, or the response is not a parseable TOON payload.
        """
        if self._client is None:
            return super().generate_cad_toon(prompt)
        try:
            text = self._generate_via_client(prompt)
            if isinstance(text, str) and toon_extended.from_toon(text.strip()):
                return text.strip()
        except Exception:
            pass
        return super().generate_cad_toon(prompt)

    def _generate_via_client(self, prompt: str) -> str:
        assert self._client is not None
        user_message = {"role": "user", "content": prompt}
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}, user_message]
        if self.provider == "claude":
            response = self._client.messages.create(
                model=self.model, max_tokens=1024, messages=[user_message]
            )
            return "".join(block.text for block in response.content if block.type == "text")
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""
