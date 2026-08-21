"""
Prepare JSONL for instruction fine-tuning from TOON-formatted data.

Expected input: a JSON file with a list of examples. Each example is a dict:
{
  "objects": [ ... ],            # list of objects to include as TOON
  "instruction": "...",       # instruction for the model
  "completion": "..."         # expected model completion for supervised fine-tuning
}

Usage:
  python prepare_finetune_from_toon.py examples.jsonl output.jsonl

This script will convert each example to a JSONL line with keys {"prompt": ..., "completion": ...}.
The prompt contains the instruction + TOON block, using toon_extended.build_prompt_for_llm.
"""

from __future__ import annotations

import json
import sys

from sdk.toon_extended import build_prompt_for_llm


def prepare_from_examples(examples: list[dict]) -> list[dict]:
    out = []
    for ex in examples:
        objs = ex.get("objects", [])
        instr = ex.get("instruction", "")
        completion = ex.get("completion", "")
        # Build prompt text using TOON with schema inferred
        prepared = build_prompt_for_llm(instr, objs, include_schema=True)
        prompt_text = prepared["prompt_text"]
        out.append({"prompt": prompt_text, "completion": completion})
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_finetune_from_toon.py examples.json output.jsonl")
        sys.exit(2)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    with open(infile, encoding="utf-8") as f:
        examples = json.load(f)
    prepared = prepare_from_examples(examples)
    with open(outfile, "w", encoding="utf-8") as out:
        for p in prepared:
            out.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prepared)} examples to {outfile}")


if __name__ == "__main__":
    main()
