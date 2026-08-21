"""
cadgenesis.inference.mcts
=========================
Test-time compute: scale inference compute instead of parameters.

Modern LLM scaling (OpenAI o1, DeepSeek-R1, Kimi K2) spends *more compute at
inference* to solve harder problems.  For a CAD-generation model the natural
verifiable signal is the design oracle, so these methods are all oracle-driven:

* :func:`best_of_n`       — sample n sequences, keep the one with the highest
                            oracle reward (simple, very effective).
* :func:`self_consistency`— majority vote over sampled sequences (Robust
                            Chain-of-Thought style), strong with temperature.
* :func:`mcts`            — genuine Monte-Carlo tree search over partial
                            sequences: sample expansions from a prefix, evaluate
                            leaves with the oracle, backprop rewards, and
                            re-search the best branch with UCT.  Process-level
                            search without a learned PRM.
"""

from __future__ import annotations

import math
from collections import Counter

from cadgenesis.distillation.rlvr import VerifiableOracle
from cadgenesis.inference.engine import CADInferenceEngine, GenerationResult


def best_of_n(
    engine: CADInferenceEngine,
    text: str,
    oracle: VerifiableOracle,
    n: int = 8,
    max_len: int = 64,
    temperature: float = 1.0,
) -> tuple[GenerationResult, float]:
    """Sample ``n`` completions and return the oracle-highest (with its reward)."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    best_result: GenerationResult | None = None
    best_reward = -1.0
    for _ in range(n):
        result = engine.sample(text, max_len=max_len, temperature=temperature)
        reward = float(oracle.verify(result.ids))
        if reward > best_reward:
            best_result, best_reward = result, reward
    assert best_result is not None
    return best_result, best_reward


def self_consistency(
    engine: CADInferenceEngine,
    text: str,
    n: int = 8,
    max_len: int = 64,
    temperature: float = 1.0,
) -> GenerationResult:
    """Majority-vote the sampled sequences by their id-token tuples."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    counts: Counter = Counter()
    results: list[GenerationResult] = []
    for _ in range(n):
        result = engine.sample(text, max_len=max_len, temperature=temperature)
        results.append(result)
        counts[tuple(result.ids)] += 1
    winner = max(counts, key=counts.get)  # type: ignore[arg-type]
    for result in results:
        if tuple(result.ids) == winner:
            return result
    return results[0]


def mcts(
    engine: CADInferenceEngine,
    text: str,
    oracle: VerifiableOracle,
    iterations: int = 8,
    max_len: int = 32,
    temperature: float = 1.0,
    branch: int = 3,
    rollout_len: int = 4,
    c: float = 1.4,
    seed_prefix: list[int] | None = None,
) -> tuple[GenerationResult, float]:
    """
    Monte-Carlo tree search over partial CAD sequences.

    The tree is keyed by id-tuples.  Each iteration:

    1. **Selection** — descend from the root by UCT (``v + c*sqrt(ln N/n)``).
    2. **Expansion** — sample ``branch`` continuations of ``rollout_len`` new
       tokens from the selected prefix with the engine.
    3. **Evaluation** — oracle reward of each complete continuation.
    4. **Backprop** — propagate mean rewards up the visited path.

    Returns the best (result, reward) found.  ``seed_prefix`` starts the search
    from a fixed partial program.
    """
    if iterations < 1 or branch < 1 or rollout_len < 1:
        raise ValueError("iterations, branch and rollout_len must be >= 1.")

    root = tuple(seed_prefix or [engine.tokenizer.bos_id])
    tree: dict[tuple, dict] = {root: {"value": 0.0, "visits": 0, "children": {}}}
    best_result: GenerationResult | None = None
    best_reward = -1.0

    def uct(node: tuple) -> tuple:
        children = tree[node]["children"]
        if not children:
            return node
        total = tree[node]["visits"]
        best_child = None
        best_score = -math.inf
        for child, stats in children.items():
            v = stats["value"]
            n = max(stats["visits"], 1)
            score = v + c * math.sqrt(math.log(total + 1.0) / n)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child or node

    for _ in range(iterations):
        node = root
        path = [root]
        # Selection: descend while the node has children with spare capacity.
        while tree[node]["children"]:
            nxt = uct(node)
            if nxt == node or nxt not in tree:
                break
            node = nxt
            path.append(node)

        # Expansion: sample continuations from this prefix.
        prefix = list(node)
        expanded = False
        for _ in range(branch):
            result = engine.sample(
                text,
                max_len=max_len,
                temperature=temperature,
                start_ids=prefix,
            )
            child_key = tuple(result.ids)
            if child_key not in tree:
                tree[child_key] = {"value": 0.0, "visits": 0, "children": {}}
                tree[node]["children"][child_key] = tree[child_key]
            reward = float(oracle.verify(result.ids))
            expanded = True
            # Backprop along the path from this child to the root.
            stats = tree[child_key]
            stats["value"] = (stats["value"] * stats["visits"] + reward) / (stats["visits"] + 1)
            stats["visits"] += 1
            for anc in reversed(path):
                a = tree[anc]
                a["value"] = (a["value"] * a["visits"] + reward) / (a["visits"] + 1)
                a["visits"] += 1
            if reward > best_reward:
                best_reward = reward
                best_result = result
        if not expanded:
            break

    assert best_result is not None
    return best_result, best_reward


__all__ = ["best_of_n", "mcts", "self_consistency"]
