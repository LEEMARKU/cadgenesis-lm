"""
cadgenesis.cli.generate
=======================
``python -m cadgenesis.cli.generate`` — run local generation.

Loads a checkpoint + tokenizer and decodes a prompt with the
:class:`CADInferenceEngine` (greedy or beam); prints tokens, confidence,
TOON output and optional per-token breakdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch

logger = logging.getLogger("cadgenesis.cli.generate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis generate", description="Generate CAD tokens from a prompt"
    )
    parser.add_argument(
        "prompt", nargs="?", default=None, help="prompt text (or use --prompt-file)"
    )
    parser.add_argument("--prompt-file", default=None, help="read prompt from a file")
    parser.add_argument("--model", default=None, help="checkpoint path (default: CADGENESIS_MODEL)")
    parser.add_argument(
        "--tokenizer", default=None, help="tokenizer directory (default: CADGENESIS_TOKENIZER)"
    )
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def load_engine(checkpoint: str, device: str | None = None):

    from cadgenesis.config import CADConfig
    from cadgenesis.inference.engine import CADInferenceEngine
    from cadgenesis.tokenizer import AutonomousCADTokenizer, restore_vocab_tokens
    from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = (
        CADConfig.from_dict(state["config"])
        if isinstance(state.get("config"), dict)
        else CADConfig.mini()
    )
    tokenizer = (
        AutonomousCADTokenizer.load(os.environ["CADGENESIS_TOKENIZER"])
        if os.environ.get("CADGENESIS_TOKENIZER")
        else AutonomousCADTokenizer.build_mini()
    )
    # A train-time checkpoint carries its vocab; re-register so ids match.
    restore_vocab_tokens(tokenizer, state.get("vocab_tokens", []))
    model = GeometryAwareTransformer(config)
    model.load_state_dict(state["model_state_dict"])
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return CADInferenceEngine(model, tokenizer, device=device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = args.prompt
    if not prompt and args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read()
    if not prompt:
        print("error: provide a prompt or --prompt-file")
        return 2
    checkpoint = args.model or os.environ.get("CADGENESIS_MODEL")
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"error: model checkpoint not found: {checkpoint}")
        return 2
    engine = load_engine(checkpoint, args.device)
    if args.beam_width > 1:
        result = engine.beam(prompt, beam_width=args.beam_width, max_len=args.max_len)
    else:
        result = engine.greedy(prompt, max_len=args.max_len)
    if args.json:
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "tokens": result.tokens,
                    "confidence": round(float(result.confidence), 6),
                    "toon": result.toon,
                    "stopped_on_eos": result.stopped_on_eos,
                }
            )
        )
    else:
        print("tokens:   ", " ".join(result.tokens))
        print(f"confidence: {result.confidence:.4f}")
        print(f"eos:        {result.stopped_on_eos}")
        print(f"toon:       {result.toon[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
