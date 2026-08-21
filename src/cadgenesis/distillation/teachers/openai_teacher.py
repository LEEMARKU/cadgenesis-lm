"""cadgenesis.distillation.teachers.openai_teacher
=================================================
OpenAI-compatible teacher adapter.

An :class:`OpenAITeacher` queries an OpenAI-compatible chat endpoint
(``gpt-4o`` by default) for a TOON payload in response to a natural
language CAD prompt.  The ``openai`` package is imported lazily behind a
try/except (same pattern as ``hf_teacher.py``): when the package is not
installed, no API key is configured, or the API call fails, the adapter
falls back to the rule-based generator of
:class:`~cadgenesis.distillation.distill_pipeline.TeacherModelInterface`,
so it never crashes in an offline environment.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.distillation.distill_pipeline import TeacherModelInterface
from sdk import toon_extended

__all__ = ["OpenAITeacher"]


def _import_openai() -> Any | None:
    """Import the OpenAI client class, or None when unavailable."""
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI


class OpenAITeacher(TeacherModelInterface):
    """OpenAI-compatible teacher adapter with rule-based fallback.

    Parameters
    ----------
    model:
        OpenAI chat model id (default ``"gpt-4o"``).
    api_key:
        Optional API key; falls back to ``OPENAI_API_KEY`` env var and to
        the rule-based generator when absent.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        super().__init__(provider="openai", api_key=api_key)
        self.model = model
        self._client: Any | None = None
        client_cls = _import_openai()
        if client_cls is not None and self.api_key:
            self._client = client_cls(api_key=self.api_key)

    def generate_cad_toon(self, prompt: str) -> str:
        """Query the OpenAI-compatible endpoint for a TOON payload.

        Falls back to the rule-based generator of the base class when the
        client is unavailable, no API key is configured, the request
        raises, or the response is not a parseable TOON payload.
        """
        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You convert natural-language CAD part descriptions "
                            "into a single TOON payload (header, optional schema and one or "
                            "more data rows). Reply with only the TOON payload.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                text = response.choices[0].message.content
                if isinstance(text, str) and toon_extended.from_toon(text.strip()):
                    return text.strip()
            except Exception:
                pass
        return super().generate_cad_toon(prompt)
