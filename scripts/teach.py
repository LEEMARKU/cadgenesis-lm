"""
teach.py
========
LLM-to-LLM teaching: use ANY instruction-tuned HuggingFace teacher model to
generate CAD training data, then train the CADGenesis-LM student on it.

The pipeline is teacher-agnostic — swap the teacher model id and the whole
loop (generate -> validate -> train) works unchanged:

    python scripts/teach.py --teacher Qwen/Qwen2.5-1.5B-Instruct \
        --prompts-file data/prompts.txt --model 1.5b \
        --epochs 50 --out-dir checkpoints/teach-qwen-1.5b

    # local model directory (e.g. your own fine-tuned teacher)
    python scripts/teach.py --teacher ./my-teacher --prompts-file data/prompts.txt

    # instant mock teacher (no download, for pipeline smoke tests)
    python scripts/teach.py --teacher mock --prompts "a steel box,a bracket"

Every run writes the teacher's identity into the checkpoint as provenance:
    checkpoint["teaching"] = {"teacher_model_id": ..., "dataset_sha256": ...}

Steps:
  1. Load teacher (HF model id or local dir; 'mock' for a fake teacher).
  2. Generate (prompt -> CAD feature tokens) records for every prompt.
  3. Drop records with empty programs (quality gate).
  4. Write the dataset to data/teacher_<model-tag>.jsonl.
  5. Train the student with train.py's pipeline (same digest/checkpoint format).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cadgenesis.distillation.teachers.hf_teacher import (  # noqa: E402
    CAD_DEFAULT_VOCAB,
    HFTeacher,
)


def _mock_cad(prompt: str) -> list[str]:
    """Deterministic mock teacher: sketch->extrude->box, +cylinder for rods."""
    lower = prompt.lower()
    program = ["SKETCH_RECT", "EXTRUDE", "BOX"]
    if "rod" in lower or "bracket" in lower or "mount" in lower:
        program.append("CYLINDER")
    if "sphere" in lower or "feet" in lower or "ball" in lower:
        program.append("SPHERE")
    if "housing" in lower or "cylinder" in lower:
        program = ["CYLINDER", "EXTRUDE", "BOX"]
    return program


class MockTeacher:
    """Drop-in teacher stand-in (no model download, pipeline smoke tests)."""

    def __init__(self, allowed_tokens=CAD_DEFAULT_VOCAB):
        self.allowed_tokens = allowed_tokens
        self.model_id = "mock"

    def generate_cad_program(self, prompt: str) -> list[str]:
        return _mock_cad(prompt)

    def generate_feature_record(self, prompt: str) -> dict[str, list[str]]:
        return {"text": prompt, "cad": self.generate_cad_program(prompt)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-to-LLM teaching: teacher generates data, student trains."
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model id / local dir of the teacher LLM, or 'mock'.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated prompts (or @file for one prompt per line).",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File with one prompt per line.",
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default=None,
        help="Where to write the teacher dataset (default data/teacher_<tag>.jsonl).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Teacher generation budget per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Teacher sampling temperature (0 = greedy).",
    )
    # ---- student training knobs (forwarded to train.py) ----
    parser.add_argument(
        "--model",
        choices=["mini", "small", "base", "1.5b", "large"],
        default="mini",
        help="Student architecture preset.",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fsdp", action="store_true", default=False)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw8bit"],
        default="adamw",
        help="Student optimizer (adamw8bit = 8-bit, for 1.5B on limited RAM).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force student training device ('cpu' or 'cuda'). Default: auto. "
        "Use 'cpu' for 1.5B-scale students on small-VRAM boxes.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/teach-run",
        help="Student checkpoint output dir.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only load the first N teacher records for training.",
    )
    return parser.parse_args()


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as fh:
            prompts.extend(line.strip() for line in fh if line.strip())
    if args.prompts:
        text = args.prompts
        if text.startswith("@"):
            with open(text[1:], encoding="utf-8") as fh:
                prompts.extend(line.strip() for line in fh if line.strip())
        else:
            prompts.extend(p.strip() for p in text.split(",") if p.strip())
    if not prompts:
        raise SystemExit("no prompts: pass --prompts or --prompts-file")
    return prompts


def _build_teacher(args: argparse.Namespace):
    if args.teacher == "mock":
        return MockTeacher(), "mock"
    teacher = HFTeacher(
        model_id=args.teacher,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"[Teacher] loading {args.teacher} ...")
    teacher.load()  # fail fast on download/load errors
    return teacher, args.teacher


def _tag(model_id: str) -> str:
    if model_id == "mock":
        return "mock"
    return model_id.replace("/", "-").replace("\\", "-")


def _generate_dataset(teacher, prompts: list[str], out_path: Path) -> Path:
    records: list[dict] = []
    rejected = 0
    t0 = time.time()
    for i, prompt in enumerate(prompts, start=1):
        rec = teacher.generate_feature_record(prompt)
        if rec["cad"]:
            records.append(rec)
        else:
            rejected += 1
        if i % 20 == 0 or i == len(prompts):
            print(
                f"[Generate] {i}/{len(prompts)} ok={len(records)} "
                f"rejected={rejected} ({time.time() - t0:.0f}s)"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Generate] {len(records)} records -> {out_path} ({rejected} rejected)")
    return out_path


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args)
    print(
        "=================================================================\n"
        "CADGenesis-LM LLM-to-LLM Teaching\n"
        f"teacher={args.teacher} | student={args.model} | "
        f"prompts={len(prompts)}\n"
        "================================================================="
    )

    teacher, teacher_id = _build_teacher(args)
    out_data = Path(args.out_data or f"data/teacher_{_tag(teacher_id)}.jsonl")
    dataset = _generate_dataset(teacher, prompts, out_data)

    # ---- train the student through train.py's pipeline ----
    import train

    train_args = argparse.Namespace(
        data=str(dataset),
        out_dir=args.out_dir,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        schedule=args.schedule,
        packed=True,
        val_fraction=args.val_fraction,
        max_records=args.max_records,
        warmup_steps=args.warmup_steps,
        resume_from=None,
        bf16=args.bf16,
        mu_transfer=False,
        fsdp=args.fsdp,
        d_model=None,
        optimizer=args.optimizer,
        device=args.device,
        teacher=None,
        prompts=None,
        prompts_file=None,
    )
    result = train.run_training(train_args)

    # ---- stamp teacher provenance onto the best checkpoint ----
    best = Path(args.out_dir) / "best_checkpoint.pt"
    if best.exists():
        import torch

        ck = torch.load(best, map_location="cpu", weights_only=False)
        ck.setdefault("teaching", {})["teacher_model_id"] = teacher_id
        ck["teaching"]["teacher_dsl_vocab"] = list(CAD_DEFAULT_VOCAB)
        torch.save(ck, best)
        print(f"[Teach] provenance stamped on {best}: teacher={teacher_id}")

    print(
        "=================================================================\n"
        f"Teaching complete: best_val={result['best_val']:.4f} "
        f"digest={result['digest'][:16]}...\n"
        "================================================================="
    )


if __name__ == "__main__":
    main()
