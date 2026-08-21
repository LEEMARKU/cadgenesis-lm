"""
Enhanced TOON utilities with typed-schema and chunking/streaming support.

Features:
- to_toon(objects, fields=None, delimiter='|', types=None, include_schema=False)
  If include_schema=True and types provided (list of strings like 'int','float','str',
  'bool'), a second line after header contains types.
- from_toon(toon_str, delimiter='|') -> list[dict] that detects and applies schema if present.
- chunk_toon(objects, chunk_size, fields=None, types=None, include_schema=False,
  delimiter='|') -> list of chunks with metadata
- stream_toon(objects, chunk_size, ...) -> generator yielding chunk dicts
  (useful for streaming endpoints)
- Integration helpers: build_prompt_for_llm(instruction, objects, ...)
- estimate_tokens uses tiktoken if available

This file is intentionally self-contained and does not modify the original toon.py.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

# Optional token estimator
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # type: ignore


# Basic escaping/unescaping similar to toon.py
def _escape_value(val: object, delimiter: str) -> str:
    s = "" if val is None else str(val)
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    if delimiter:
        s = s.replace(delimiter, f"\\{delimiter}")
    return s


def _unescape_value(s: str, delimiter: str) -> str:
    res_chars = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 1
            if i >= len(s):
                res_chars.append("\\")
                break
            nxt = s[i]
            if nxt == "n":
                res_chars.append("\n")
            else:
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


# Type casting utilities
def _cast_value(s: str, typ: str | None) -> Any:
    if typ is None:
        return s
    # treat empty string as None for non-string types
    if s == "" and typ != "str":
        return None
    try:
        t = typ.lower()
        if t == "int" or t == "integer":
            return int(s)
        if t == "float" or t == "double" or t == "number":
            return float(s)
        if t == "bool" or t == "boolean":
            ls = s.lower()
            if ls in ("true", "1", "yes", "y"):
                return True
            if ls in ("false", "0", "no", "n"):
                return False
            # fallback
            return bool(s)
        if t == "str" or t == "string":
            return s
        # Unknown type: return string
        return s
    except Exception:
        # If casting fails, return original string
        return s


def to_toon(
    objects: list[dict],
    fields: list[str] | None = None,
    delimiter: str = "|",
    types: list[str] | None = None,
    include_schema: bool = False,
) -> str:
    if objects is None:
        raise ValueError("objects must be a list")
    if len(objects) == 0:
        return ""
    if fields is None:
        fields = list(objects[0].keys())
    header = delimiter.join(fields)
    schema_line = ""
    if include_schema:
        if types is None:
            # infer basic types from first object
            types = []
            for f in fields:
                v = objects[0].get(f, None)
                if v is None:
                    types.append("str")
                elif isinstance(v, bool):
                    types.append("bool")
                elif isinstance(v, int) and not isinstance(v, bool):
                    types.append("int")
                elif isinstance(v, float):
                    types.append("float")
                else:
                    types.append("str")
        schema_line = delimiter.join(types)

    rows: list[str] = []
    for obj in objects:
        row_values: list[str] = []
        for f in fields:
            val = obj.get(f, "")
            row_values.append(_escape_value(val, delimiter))
        rows.append(delimiter.join(row_values))

    out_lines = [header]
    if include_schema:
        out_lines.append(schema_line)
    out_lines.extend(rows)
    return "\n".join(out_lines)


def from_toon(toon_str: str, delimiter: str = "|") -> list[dict]:
    if not toon_str:
        return []
    lines = toon_str.splitlines()
    if not lines:
        return []
    header_line = lines[0]
    fields = header_line.split(delimiter) if header_line else []
    types: list[str] | None = None
    start_idx = 1
    # Detect schema if second line contains only type tokens
    if len(lines) > 1:
        cand = lines[1]
        cand_parts = cand.split(delimiter)
        # simple heuristic: each part matches common type names
        common = set(
            ["int", "integer", "float", "double", "number", "str", "string", "bool", "boolean"]
        )
        if all(p.lower() in common for p in cand_parts):
            types = cand_parts
            start_idx = 2
    objs: list[dict] = []
    for line in lines[start_idx:]:
        parts = _split_escaped(line, delimiter)
        values = [_unescape_value(p, delimiter) for p in parts]
        while len(values) < len(fields):
            values.append("")
        obj: dict[str, Any] = {}
        for i, f in enumerate(fields):
            typ = types[i] if types and i < len(types) else None
            obj[f] = _cast_value(values[i], typ)
        objs.append(obj)
    return objs


def estimate_tokens(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model or "gpt-4")
            return len(enc.encode(text))
        except Exception:
            pass
    word_count = len(text.split())
    return int(word_count * 1.3) + 1


def compare_toon_json_tokens(
    objects: list[dict],
    fields: list[str] | None = None,
    model: str | None = None,
    delimiter: str = "|",
    types: list[str] | None = None,
    include_schema: bool = False,
) -> dict:
    json_text = json.dumps(objects, ensure_ascii=False)
    toon_text = to_toon(
        objects, fields=fields, delimiter=delimiter, types=types, include_schema=include_schema
    )
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


def chunk_toon(
    objects: list[dict],
    chunk_size: int = 100,
    fields: list[str] | None = None,
    types: list[str] | None = None,
    include_schema: bool = False,
    delimiter: str = "|",
) -> list[dict]:
    """Split objects into multiple TOON chunks.

    Returns a list of dicts: { 'chunk_index': i, 'total': n, 'toon': toon_text }
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    total = (len(objects) + chunk_size - 1) // chunk_size
    chunks: list[dict] = []
    for i in range(total):
        start = i * chunk_size
        end = min(start + chunk_size, len(objects))
        subset = objects[start:end]
        toon_text = to_toon(
            subset,
            fields=fields,
            delimiter=delimiter,
            types=types,
            include_schema=(include_schema and i == 0),
        )
        # include schema only in first chunk if requested
        chunks.append(
            {"chunk_index": i, "total": total, "start": start, "end": end, "toon": toon_text}
        )
    return chunks


def stream_toon(
    objects: list[dict],
    chunk_size: int = 100,
    fields: list[str] | None = None,
    types: list[str] | None = None,
    include_schema: bool = False,
    delimiter: str = "|",
) -> Generator[dict, None, None]:
    chunks = chunk_toon(
        objects,
        chunk_size=chunk_size,
        fields=fields,
        types=types,
        include_schema=include_schema,
        delimiter=delimiter,
    )
    yield from chunks


def build_prompt_for_llm(
    instruction: str,
    objects: list[dict],
    fields: list[str] | None = None,
    types: list[str] | None = None,
    include_schema: bool = False,
    delimiter: str = "|",
) -> dict:
    """Return a dict containing the prompt text and token estimates useful for sending to an LLM.

    The prompt_text includes a brief explanation header plus the TOON block and the instruction.
    """
    toon_text = to_toon(
        objects, fields=fields, delimiter=delimiter, types=types, include_schema=include_schema
    )
    prompt_text = f"Data (TOON):\n{toon_text}\n\nInstructions:\n{instruction}"
    tokens = estimate_tokens(prompt_text)
    return {"prompt_text": prompt_text, "prompt_tokens_estimate": tokens}


# quick smoke test when run directly
if __name__ == "__main__":
    sample = [
        {"id": 1, "name": "Widget A", "price": 9.99, "active": True},
        {"id": 2, "name": "Widget B", "price": 15.5, "active": False},
    ]
    toon = to_toon(sample, include_schema=True)
    print("TOON with schema:\n", toon)
    parsed = from_toon(toon)
    print("Parsed back:", parsed)
    chunks = chunk_toon(sample * 3, chunk_size=2, include_schema=True)
    print("Chunks:", len(chunks))
