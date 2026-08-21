"""cadgenesis.tokenizer.engineering
===================================
Engineering-notation normalisation for the CAD tokenizer.

Maps CAD/engineering notation found in natural-language prompts onto plain
text the language tokenizer can segment, without losing meaning:

* ``Ø25``         -> ``diameter 25 mm``
* ``R12.5``       -> ``radius 12.5 mm``
* ``M8x1.25``     -> ``thread M8 pitch 1.25 mm``
* ``±0.02``       -> ``tolerance plus or minus 0.02 mm``
* ``-12.75``      -> ``minus 12.75``
* ``1e-5``        -> ``0.00001``
* ``25 mm``/``in``/``deg``/``rad`` -> expanded, unambiguous units
* ``(10,20,30)``  -> ``coordinate x 10 y 20 z 30``
* ``[0,0,1]``     -> ``vector x 0 y 0 z 1``

The transform is deterministic, idempotent, and a no-op for plain text.
"""

from __future__ import annotations

import re

__all__ = ["normalize_engineering_notation", "parse_engineering_tokens"]

_EXPONENT_RE = re.compile(r"(?<![A-Za-z0-9_])([-+]?[0-9]*\.?[0-9]+)[eE]([-+]?[0-9]+)")

_DIAMETER_RE = re.compile(r"Ø\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:mm|cm|m|inch|in))?")
_RADIUS_RE = re.compile(r"(?<![A-Za-z0-9_])R\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:mm|cm|m|inch|in))?")
_THREAD_RE = re.compile(
    r"(?<![A-Za-z0-9_])M\s*([0-9]+(?:\.[0-9]+)?)\s*[xX\u00d7]\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*(?:mm|cm|m|inch|in))?"
)
_TOLERANCE_RE = re.compile(r"±\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:mm|cm|m|inch|in))?")

_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([0-9]+(?:\.[0-9]+)?)\s*"
    r"(mm|cm|m|inch|in|deg|rad)\b"
)
_MINUS_RE = re.compile(r"(?<![A-Za-z0-9_.])-([0-9]+(?:\.[0-9]+)?)")
_PLUS_RE = re.compile(r"(?<![A-Za-z0-9_.])\+([0-9]+(?:\.[0-9]+)?)")

_COORD_RE = re.compile(r"\(([-0-9]+(?:\.[0-9]+)?(?:,\s*[-0-9]+(?:\.[0-9]+)?)*)\)")
_VECTOR_RE = re.compile(r"\[([-0-9]+(?:\.[0-9]+)?(?:,\s*[-0-9]+(?:\.[0-9]+)?)*)\]")


def _expand_exponent(match: re.Match) -> str:
    mantissa, exponent = match.group(1), int(match.group(2))
    value = float(mantissa) * (10.0**exponent)
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _diameter(match: re.Match) -> str:
    return f"diameter {match.group(1)} mm"


def _radius(match: re.Match) -> str:
    return f"radius {match.group(1)} mm"


def _thread(match: re.Match) -> str:
    return f"thread M{match.group(1)} pitch {match.group(2)} mm"


def _tolerance(match: re.Match) -> str:
    return f"tolerance plus or minus {match.group(1)} mm"


def _unit(match: re.Match) -> str:
    return f"{match.group(1)} {match.group(2)}"


def _minus(match: re.Match) -> str:
    return f"minus {match.group(1)}"


def _plus(match: re.Match) -> str:
    return f"plus {match.group(1)}"


def _coord(match: re.Match, kind: str) -> str:
    parts = [p.strip() for p in match.group(1).split(",")]
    names = ["x", "y", "z"]

    def _name(i: int) -> str:
        if i < len(names):
            return names[i]
        return f"a{i - len(names) + 1}"

    body = " ".join(f"{_name(i)} {p}" for i, p in enumerate(parts))
    return f"{kind} {body}"


def _coords(match: re.Match) -> str:
    return _coord(match, "coordinate")


def _vector(match: re.Match) -> str:
    return _coord(match, "vector")


def normalize_engineering_notation(text: str) -> str:
    """Deterministically rewrite engineering notation into plain text.

    Idempotent: ``normalize(normalize(text)) == normalize(text)``.
    Plain text (no notation) passes through unchanged.
    """
    if not text:
        return text
    result = _EXPONENT_RE.sub(_expand_exponent, text)
    result = _DIAMETER_RE.sub(_diameter, result)
    result = _RADIUS_RE.sub(_radius, result)
    result = _THREAD_RE.sub(_thread, result)
    result = _TOLERANCE_RE.sub(_tolerance, result)
    result = _UNIT_RE.sub(_unit, result)
    result = _PLUS_RE.sub(_plus, result)
    result = _MINUS_RE.sub(_minus, result)
    result = _COORD_RE.sub(_coords, result)
    result = _VECTOR_RE.sub(_vector, result)
    return result


def parse_engineering_tokens(text: str) -> list[str]:
    """Tokenise a prompt after engineering-notation normalisation.

    Splits on whitespace and strips punctuation so the result is ready for
    word-level language tokenization.
    """
    normalized = normalize_engineering_notation(text)
    keep = r"[^\w.+x\u00d7-]"
    return [tok for tok in re.split(r"\s+", re.sub(keep, " ", normalized)) if tok]
