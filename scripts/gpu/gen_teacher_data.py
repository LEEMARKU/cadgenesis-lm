"""Generate a DeepSeek-R1 teacher CAD dataset on a GPU box.

Each prompt produces a ``{"text", "cad": [feature-token strings]}`` record the
same shape as ``data/cad_programs.jsonl``, so the output is a drop-in
``--data`` source for ``train.py`` (the tokenizer registers any new feature
names automatically).

    python scripts/gpu/gen_teacher_data.py --prompts data/prompts.txt \
        --out data/teacher_deepseek.jsonl --device cuda

DeepSeek-R1-Distill-Qwen-1.5B downloads (~3 GB) on first run.  ~30 s per
program on CPU, well under a second per program on a modern GPU.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cadgenesis.adapters.deepseek_r1 import (
    DeepSeekR1DataGenerator,
    DeepSeekR1Reasoner,
    DeepSeekR1Teacher,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="file with one prompt per line")
    parser.add_argument("--out", default="data/teacher_deepseek.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {args.prompts}")

    reasoner = DeepSeekR1Reasoner(
        device=args.device,
        torch_dtype=torch.bfloat16,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    generator = DeepSeekR1DataGenerator(DeepSeekR1Teacher(reasoner), vocab=None, verbose=True)
    records = generator.generate_feature_records(prompts, reasoning=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
