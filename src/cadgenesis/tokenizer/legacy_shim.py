"""
cadgenesis.tokenizer.legacy_shim
==================================
Backward-compatibility bridge between the original data.py and the new
AutonomousCADTokenizer.

Purpose
-------
Existing code (train.py, generate.py, Colab notebook) imports symbols
directly from data.py:

    from data import (LangTokenizer, build_dataset, CAD_TOK2ID, PAD_ID,
                      BOS_ID, EOS_ID, cad_token_type, CAD_VOCAB)

This shim makes those same symbols available from the new package so
existing code continues to work without modification.

Usage
-----
Replace the old import line:

    # OLD
    from data import LangTokenizer, build_dataset, CAD_TOK2ID, ...

    # NEW (drop-in)
    from cadgenesis.tokenizer.legacy_shim import (
        LangTokenizer, build_dataset, CAD_TOK2ID, CAD_ID2TOK,
        PAD_ID, BOS_ID, EOS_ID, CAD_VOCAB, cad_token_type,
        value_to_bin, sample_value, NUM_BINS,
        get_tokenizer,
    )

The shim exposes a module-level singleton ``AutonomousCADTokenizer``
in mini mode, so the numeric and CAD vocabularies are identical to the
original data.py definitions.
"""

from __future__ import annotations

import random

from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.language import LegacyWordTokenizer

# ---------------------------------------------------------------------------
# Module-level singleton (mini tokenizer, legacy 20-bin quantization)
# ---------------------------------------------------------------------------

_MINI_TOK: AutonomousCADTokenizer | None = None


def get_tokenizer() -> AutonomousCADTokenizer:
    """Return the module-level mini tokenizer singleton."""
    global _MINI_TOK
    if _MINI_TOK is None:
        _MINI_TOK = AutonomousCADTokenizer.build_mini()
    return _MINI_TOK


# ---------------------------------------------------------------------------
# Legacy constants (match data.py exactly)
# ---------------------------------------------------------------------------

NUM_BINS = [round(0.5 + 0.5 * i, 2) for i in range(20)]


def value_to_bin(v: float) -> int:
    return min(range(len(NUM_BINS)), key=lambda i: abs(NUM_BINS[i] - v))


def sample_value() -> float:
    return random.choice(NUM_BINS)


# These are computed lazily from the singleton so they remain consistent
# with data.py's flat dicts.
def _get_cad_vocab_list() -> list[str]:
    tok = get_tokenizer()
    return list(tok.vocab.to_tok2id().keys())


# Expose the same names as data.py
def _lazy_tok2id():
    return get_tokenizer().vocab.to_tok2id()


def _lazy_id2tok():
    return get_tokenizer().vocab.to_id2tok()


# The mini tokenizer has the legacy primitives at fixed IDs matching data.py:
#   <pad>=0  <bos>=1  <eos>=2  <unk>=3  …  BOX  CYLINDER  SPHERE  SKETCH_RECT
#   EXTRUDE  NUM_0..NUM_19
# This is guaranteed by AutonomousCADTokenizer.build_mini().

CAD_VOCAB = [r.token_str for r in get_tokenizer().vocab]
CAD_TOK2ID = get_tokenizer().vocab.to_tok2id()
CAD_ID2TOK = get_tokenizer().vocab.to_id2tok()
PAD_ID = get_tokenizer().pad_id
BOS_ID = get_tokenizer().bos_id
EOS_ID = get_tokenizer().eos_id


def cad_token_type(tok_id: int) -> int:
    """Return the integer type embedding id for a CAD token id."""
    tok = get_tokenizer()
    if tok_id in tok.vocab:
        return tok.vocab.type_id_of(tok_id)
    return 0  # SPECIAL family


# ---------------------------------------------------------------------------
# LangTokenizer — drop-in replacement for data.py's LangTokenizer
# ---------------------------------------------------------------------------


class LangTokenizer(LegacyWordTokenizer):
    """
    Backward-compatible LangTokenizer alias.

    Identical to data.py's LangTokenizer but inherits all improvements
    from LegacyWordTokenizer (decode, save/load, etc.).
    """

    pass


# ---------------------------------------------------------------------------
# Dataset generation (mirrors data.py's build_dataset / generators)
# ---------------------------------------------------------------------------


def generate_example() -> tuple[str, list[int]]:
    """Mirror of data.py's generate_example()."""
    import random

    generators = [gen_box, gen_cylinder, gen_sphere, gen_extrude]
    text, seq = random.choice(generators)()
    cad_ids = [BOS_ID] + [CAD_TOK2ID[t] for t in seq] + [EOS_ID]
    return text, cad_ids


def gen_box() -> tuple[str, list[str]]:
    w, h, d = sample_value(), sample_value(), sample_value()
    templates = [
        "Create a box that is {w} units wide, {h} units tall, and {d} units deep.",
        "Design a rectangular block with width {w}, height {h}, and depth {d}.",
        "I need a box sized {w} by {h} by {d}.",
        "Make a solid box, width {w}, height {h}, depth {d}.",
    ]
    text = random.choice(templates).format(w=w, h=h, d=d)
    seq = ["BOX", f"NUM_{value_to_bin(w)}", f"NUM_{value_to_bin(h)}", f"NUM_{value_to_bin(d)}"]
    return text, seq


def gen_cylinder() -> tuple[str, list[str]]:
    r, h = sample_value(), sample_value()
    templates = [
        "Create a cylinder with radius {r} and height {h}.",
        "Make a cylindrical part {h} units tall with a radius of {r}.",
        "Design a cylinder, radius {r}, height {h}.",
    ]
    text = random.choice(templates).format(r=r, h=h)
    seq = ["CYLINDER", f"NUM_{value_to_bin(r)}", f"NUM_{value_to_bin(h)}"]
    return text, seq


def gen_sphere() -> tuple[str, list[str]]:
    r = sample_value()
    templates = [
        "Create a sphere with radius {r}.",
        "Design a ball of radius {r}.",
        "Make a spherical part with radius {r}.",
    ]
    text = random.choice(templates).format(r=r)
    seq = ["SPHERE", f"NUM_{value_to_bin(r)}"]
    return text, seq


def gen_extrude() -> tuple[str, list[str]]:
    w, h, depth = sample_value(), sample_value(), sample_value()
    templates = [
        "Sketch a rectangle {w} by {h} and extrude it {depth} units.",
        "Create an extruded block from a {w} by {h} rectangle, extruded {depth} units deep.",
        "Draw a {w} by {h} rectangle and extrude {depth} units.",
    ]
    text = random.choice(templates).format(w=w, h=h, depth=depth)
    seq = [
        "SKETCH_RECT",
        f"NUM_{value_to_bin(w)}",
        f"NUM_{value_to_bin(h)}",
        "EXTRUDE",
        f"NUM_{value_to_bin(depth)}",
    ]
    return text, seq


def build_dataset(
    n: int,
    lang_tok: LangTokenizer | None = None,
) -> list[tuple[str, list[int]]]:
    """
    Mirror of data.py's build_dataset().

    If lang_tok has no vocab yet (len <= 2), builds vocab from the generated
    texts — identical behaviour to data.py.
    """
    raw = [generate_example() for _ in range(n)]
    texts = [t for t, _ in raw]
    if lang_tok is not None and len(lang_tok) <= 2:
        lang_tok.build_vocab(texts)
    return raw
