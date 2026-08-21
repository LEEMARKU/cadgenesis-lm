"""
TOON (Token-Oriented Object Notation) utilities

Provides:
- to_toon(objects, fields=None, delimiter='|') -> str
- from_toon(toon_str, delimiter='|') -> list[dict]
- estimate_tokens(text, model=None) -> int (uses tiktoken when available)
- compare_toon_json_tokens(objects, fields=None, model=None) -> dict
- prompt_with_toon(prompt, objects, fields=None, delimiter='|') -> str

Design choices:
- Delimiter default is '|' (pipe) for readability. Values are escaped for
  backslash, newline, and delimiter.
- Fields order is taken from first object when not provided.
- Token estimation uses tiktoken if installed; otherwise falls back to a
  conservative whitespace-based estimate.

Usage example in toon_demo.py
"""

from __future__ import annotations

import json

# Optional: try to import tiktoken for accurate token counts. If unavailable,
# fall back to heuristic.
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore


def _escape_value(val: object, delimiter: str) -> str:
    s = "" if val is None else str(val)
    # Escape backslash first
    s = s.replace("\\", "\\\\")
    # Escape newline
    s = s.replace("\n", "\\n")
    # Escape delimiter
    if delimiter:
        s = s.replace(delimiter, f"\\{delimiter}")
    return s


def _unescape_value(s: str, delimiter: str) -> str:
    # Reverse of _escape_value
    res_chars = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 1
            if i >= len(s):
                # trailing backslash, keep it
                res_chars.append("\\")
                break
            nxt = s[i]
            if nxt == "n":
                res_chars.append("\n")
            else:
                # could be escaped delimiter or backslash or other char
                res_chars.append(nxt)
            i += 1
        else:
            res_chars.append(c)
            i += 1
    return "".join(res_chars)


def _split_escaped(line: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    cur_chars: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            # include escape and next char as literal sequence to be unescaped later
            if i + 1 < len(line):
                cur_chars.append("\\")
                cur_chars.append(line[i + 1])
                i += 2
            else:
                cur_chars.append("\\")
                i += 1
        elif delimiter and c == delimiter:
            parts.append("".join(cur_chars))
            cur_chars = []
            i += 1
        else:
            cur_chars.append(c)
            i += 1
    parts.append("".join(cur_chars))
    return parts


def to_toon(objects: list[dict], fields: list[str] | None = None, delimiter: str = "|") -> str:
    """Serialize a list of dict-like objects to TOON format.

    - If fields is not provided, fields are inferred from the first object in order of keys.
    - Values are escaped for backslash, newline, and delimiter.
    """
    if objects is None:
        raise ValueError("objects must be a list")
    if len(objects) == 0:
        return ""

    if fields is None:
        # preserve insertion order (Python 3.7+ dict preserves insertion order)
        fields = list(objects[0].keys())

    header = delimiter.join(fields)
    rows: list[str] = []
    for obj in objects:
        row_values: list[str] = []
        for f in fields:
            val = obj.get(f, "")
            row_values.append(_escape_value(val, delimiter))
        rows.append(delimiter.join(row_values))

    return header + "\n" + "\n".join(rows)


def from_toon(toon_str: str, delimiter: str = "|") -> list[dict]:
    """Parse a TOON string back into list of dicts.

    Returns empty list for empty strings.
    """
    if not toon_str:
        return []
    lines = toon_str.splitlines()
    if len(lines) == 0:
        return []
    header_line = lines[0]
    fields = header_line.split(delimiter) if header_line else []
    objs: list[dict] = []
    for line in lines[1:]:
        parts = _split_escaped(line, delimiter)
        values = [_unescape_value(p, delimiter) for p in parts]
        # If row has fewer parts than fields, pad with empty string
        while len(values) < len(fields):
            values.append("")
        obj = {fields[i]: values[i] for i in range(len(fields))}
        objs.append(obj)
    return objs


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count for a string for a given model.

    If tiktoken is available, use it. Otherwise fall back to heuristic (words * 1.3).
    """
    if not text:
        return 0
    if tiktoken is not None:
        try:
            # Map some common model names to tiktoken encodings conservatively
            if model is None:
                enc = tiktoken.encoding_for_model("gpt-4")
            else:
                # If specific model not found this may raise; catch and fallback
                enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # fallback below
            pass
    # conservative heuristic: average 1.3 tokens per word
    word_count = len(text.split())
    return int(word_count * 1.3) + 1


def compare_toon_json_tokens(
    objects: list[dict],
    fields: list[str] | None = None,
    model: str | None = None,
    delimiter: str = "|",
) -> dict:
    """Return comparison of token counts between JSON and TOON.

    Returns dict with keys: json_text, toon_text, json_tokens, toon_tokens,
    savings_tokens, savings_percent
    """
    json_text = json.dumps(objects, ensure_ascii=False)
    toon_text = to_toon(objects, fields=fields, delimiter=delimiter)
    json_tokens = estimate_tokens(json_text, model=model)
    toon_tokens = estimate_tokens(toon_text, model=model)
    savings = json_tokens - toon_tokens
    pct = (savings / json_tokens * 100) if json_tokens > 0 else 0.0
    return {
        "json_text": json_text,
        "toon_text": toon_text,
        "json_tokens": json_tokens,
        "toon_tokens": toon_tokens,
        "savings_tokens": savings,
        "savings_percent": round(pct, 2),
    }


def prompt_with_toon(
    prompt: str, objects: list[dict], fields: list[str] | None = None, delimiter: str = "|"
) -> str:
    """Prepend TOON-formatted data to a prompt, separated by two newlines.

    Useful helper when sending data + instructions to an LLM. Keeps the format
    compact while leaving instructions readable.
    """
    toon_text = to_toon(objects, fields=fields, delimiter=delimiter)
    if toon_text:
        return f"Data (TOON):\n{toon_text}\n\nInstructions:\n{prompt}"
    else:
        return prompt


if __name__ == "__main__":
    # Quick smoke demo when run directly
    sample = [
        {"id": 1, "name": "Widget A", "price": 9.99, "description": "Small widget\n2 colors"},
        {"id": 2, "name": "Widget B", "price": 15.5, "description": "Large | heavy"},
    ]
    cmp = compare_toon_json_tokens(sample, model=None)
    print("JSON tokens approx:", cmp["json_tokens"])
    print("TOON tokens approx:", cmp["toon_tokens"])
    print("Savings:", cmp["savings_tokens"], f"({cmp['savings_percent']}%)")
    print("\nTOON representation:\n")
    print(cmp["toon_text"])
