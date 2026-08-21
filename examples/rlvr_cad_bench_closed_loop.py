"""
examples/rlvr_cad_bench_closed_loop.py
======================================
Closed-loop demonstration: RLVR post-training → verifiable CAD evaluation →
test-time compute → EAGLE speculative decoding, sharing ONE oracle.

Everything is wired through :class:`cadgenesis.training.RLVRPipeline`.  The
same verifier that scores the RLVR rollouts is used by the CAD benchmark and
by best-of-n / MCTS search, so every stage answers the same question: "did
the model generate valid CAD?"

Run:  python examples/rlvr_cad_bench_closed_loop.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.training.rlvr_pipeline import RLVRPipeline

torch.manual_seed(0)

PROMPTS = [
    "create a steel box",
    "make a mounting bracket",
    "design a cylindrical housing",
]

# A reference CAD program (sketch -> extrude -> box) used for the supervised
# warm-start and the EAGLE draft-head training.
REFERENCE_PROGRAM = ["SKETCH_RECT", "EXTRUDE", "BOX", "<eos>"]


def warm_start(pipe: RLVRPipeline, seq: list[int], steps: int = 100, lr: float = 5e-4) -> None:
    """Tiny supervised warm-start so the random-init policy can actually emit
    the rewarded feature tokens (standard SFT-then-RLVR ordering)."""
    tok = pipe.tokenizer
    full = [tok.bos_id, *seq]
    tgt = torch.tensor([full])
    types = torch.tensor([[_type(tok, i) for i in full]])
    opt = torch.optim.AdamW(pipe.model.parameters(), lr=lr)
    pipe.model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits, _ = pipe.model(tgt, tgt, types)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tgt[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()
    pipe.model.eval()


def _type(tok, i: int) -> int:
    try:
        return tok.vocab.type_id_of(int(i))
    except KeyError:
        return 0


def build_pipeline() -> tuple[RLVRPipeline, AutonomousCADTokenizer]:
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(PROMPTS)

    def completion_to_design(ids: list[int]) -> dict | None:
        """A completion only counts as a part when it contains a BOX feature —
        a discriminating, *verifiable* signal the RLVR loop can actually
        optimise (random completions score 0)."""
        if tok.vocab["BOX"] not in ids:
            return None
        return {
            "name": "generated_part",
            "features": [{"type": "box", "width_m": 0.1, "height_m": 0.1, "depth_m": 0.1}],
        }

    # The DesignOracle feeds each completion to the real CAD execution engine
    # and rewards +0.7 for valid geometry, +0.3 for manufacturability.
    pipe = RLVRPipeline.from_config(
        CADConfig.mini(),
        tok,
        device="cpu",
        completion_to_design=completion_to_design,
        lr=1e-4,
        kl_coef=0.01,
        num_generations=4,
        max_gen_len=12,
        format_bonus=0.1,  # reward sequences that terminate on EOS
    )
    return pipe, tok


def main() -> None:
    pipe, tok = build_pipeline()

    print("=== baseline (untrained) ===")
    before = pipe.evaluate(PROMPTS, temperature=1.0)
    print("  " + before.summary())

    # SFT warm-start on the reference program so the policy can emit the
    # rewarded feature tokens at all (random init almost never samples them).
    reference = [tok.vocab[t] for t in REFERENCE_PROGRAM]
    warm_start(pipe, reference, steps=100)
    print("=== after supervised warm-start ===")
    warmed = pipe.evaluate(PROMPTS, temperature=1.0)
    print("  " + warmed.summary())

    print("=== RLVR post-training (3 steps) ===")
    for i in range(3):
        stats = pipe.train(PROMPTS, steps=1, temperature=1.0)
        print(
            f"  step {i + 1}: "
            f"loss={stats['loss']:.3f} reward={stats['mean_reward']:.3f} "
            f"kl={stats['mean_kl']:.3f}"
        )
    after = pipe.evaluate(PROMPTS, temperature=1.0)
    print("  " + after.summary())

    print("=== test-time compute (oracle-driven) ===")
    prompt = PROMPTS[0]
    best, best_reward = pipe.best_of_n(prompt, n=4)
    print(f"  best_of_n reward:        {best_reward:.2f} ({len(best.ids)} tokens)")
    sc = pipe.self_consistency(prompt, n=4)
    print(f"  self_consistency ids:    {len(sc.ids)} tokens")
    mcts_result, mcts_reward = pipe.mcts(prompt, iterations=4, branch=2, rollout_len=2)
    print(f"  mcts reward:             {mcts_reward:.2f} ({len(mcts_result.ids)} tokens)")

    print("=== EAGLE speculative decoding (greedy-preserving) ===")
    # The reference program (already used for the warm-start) teaches the
    # draft head to predict it from the policy's hidden states.
    head = pipe.train_eagle([reference], steps=20, lr=1e-3)
    greedy = pipe.engine.greedy(prompt, max_len=12, use_cache=True)
    speculative = pipe.speculative(prompt, head, max_len=12, num_speculative_tokens=3)
    print(f"  greedy == speculative:   {greedy.ids == speculative.ids}")


if __name__ == "__main__":
    main()
