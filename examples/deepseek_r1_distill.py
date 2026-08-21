"""
examples/deepseek_r1_distill.py
===============================
Use the real open-weights reasoning model ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B``
as the teacher to generate CAD programs, then SFT + RLVR train the CADGenesis-LM
policy on those programs.

Pipeline shown:
  DeepSeek-R1 (reasoner/teacher) --(prompt)--> CAD feature program (token ids)
      --(quality filter / parse)--> (prompt, program) training pairs
      --(SFT warm-start + RLVR)--> CAD policy  --(oracle)--> verified designs

Run (real model, ~3 GB download on first use):
  python examples/deepseek_r1_distill.py --real
Run (instant, uses a fake teacher so the script works anywhere):
  python examples/deepseek_r1_distill.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.training.rlvr_pipeline import RLVRPipeline

PROMPTS = [
    "create a steel box",
    "make a mounting bracket",
    "design a cylindrical housing",
]


class _MockTeacher:
    """Fake teacher used when --real is not passed (no 3 GB download)."""

    def generate_cad_program(self, prompt: str, vocab):
        spec = "SKETCH_RECT, EXTRUDE, BOX"
        tok2id = vocab.to_tok2id()
        return [tok2id[t] for t in spec.split(", ") if t in tok2id], spec

    def generate_reasoning(self, prompt: str) -> str:
        return f"[mock] think about: {prompt} -> sketch, extrude, box"


def make_teacher(use_real: bool):
    if not use_real:
        print("using MOCK teacher (pass --real for the actual DeepSeek-R1 model)")
        return _MockTeacher()
    from cadgenesis.adapters.deepseek_r1 import DeepSeekR1Reasoner, DeepSeekR1Teacher

    reasoner = DeepSeekR1Reasoner(
        device="cpu", torch_dtype=torch.bfloat16, max_new_tokens=64, temperature=0.7
    )
    print("loading DeepSeek-R1-Distill-Qwen-1.5B (first run downloads ~3 GB)...")
    return DeepSeekR1Teacher(reasoner)


def warm_start_many(
    pipe: RLVRPipeline, seqs: list[list[int]], steps: int = 60, lr: float = 5e-4
) -> None:
    """Batched SFT on the teacher's programs so the policy can emit them."""
    if not seqs:
        return
    tok = pipe.tokenizer
    pad = int(tok.vocab["<pad>"])
    full = [[tok.bos_id, *s, tok.eos_id] for s in seqs]
    max_len = max(len(s) for s in full)
    padded = torch.full((len(full), max_len), pad, dtype=torch.long)
    for i, s in enumerate(full):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    types = torch.tensor([[_type(tok, i) for i in row] for row in padded.tolist()])
    opt = torch.optim.AdamW(pipe.model.parameters(), lr=lr)
    pipe.model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits, _ = pipe.model(padded, padded, types)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            padded[:, 1:].reshape(-1),
            ignore_index=pad,
        )
        loss.backward()
        opt.step()
    pipe.model.eval()


def _type(tok, i: int) -> int:
    try:
        return tok.vocab.type_id_of(int(i))
    except KeyError:
        return 0


def build_pipeline(tok: AutonomousCADTokenizer) -> RLVRPipeline:
    def completion_to_design(ids: list[int]) -> dict | None:
        if tok.vocab["BOX"] not in ids:
            return None
        return {
            "name": "generated_part",
            "features": [{"type": "box", "width_m": 0.1, "height_m": 0.1, "depth_m": 0.1}],
        }

    return RLVRPipeline.from_config(
        CADConfig.mini(),
        tok,
        device="cpu",
        completion_to_design=completion_to_design,
        lr=1e-4,
        kl_coef=0.01,
        num_generations=4,
        max_gen_len=12,
        format_bonus=0.1,
    )


def main() -> None:
    use_real = "--real" in sys.argv
    torch.manual_seed(0)

    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(PROMPTS)
    vocab = tok.vocab

    teacher = make_teacher(use_real)
    print("\n=== teacher distillation loop ===")
    programs: list[list[int]] = []
    for prompt in PROMPTS:
        program, spec = teacher.generate_cad_program(prompt, vocab)
        trace = teacher.generate_reasoning(prompt)
        valid = bool(program)
        print(f"  prompt    : {prompt!r}")
        print(f"  reasoning : {trace[:90]!r}...")
        print(f"  spec      : {spec[:70]!r}  ->  {program}  valid={valid}")
        if valid:
            programs.append(program)

    print("\n=== train CAD policy on teacher programs (SFT + RLVR) ===")
    pipe = build_pipeline(tok)
    before = pipe.evaluate(PROMPTS, temperature=1.0)
    print("  baseline: " + before.summary())

    warm_start_many(pipe, programs, steps=60)
    print("  after SFT on teacher programs:")
    warmed = pipe.evaluate(PROMPTS, temperature=1.0)
    print("  " + warmed.summary())

    print("  RLVR reinforcement (2 steps):")
    for i in range(2):
        stats = pipe.train(PROMPTS, steps=1, temperature=1.0)
        print(f"    step {i + 1}: loss={stats['loss']:.3f} reward={stats['mean_reward']:.3f}")

    print("\n=== oracle-verified best-of-n ===")
    for prompt in PROMPTS:
        best, reward = pipe.best_of_n(prompt, n=4)
        print(f"  {prompt!r}: reward={reward:.2f} ({len(best.ids)} tokens)")


if __name__ == "__main__":
    main()
