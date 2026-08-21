"""
CADGenesis-Mini: hybrid tokenizer + synthetic CAD dataset.

This is a deliberately small, honest, working slice of the "hybrid tokenizer"
concept from the Phase 1 architecture doc: a LANGUAGE token stream and a
GEOMETRY/FEATURE token stream, generated from paired (text, CAD-sequence)
data.

CAD objects are represented the way real CAD-generation research does it
(see DeepCAD / SkexGen-style sequence representations): a short sequence of
primitive/feature tokens interleaved with *quantized* numeric parameter
tokens, rather than continuous CAD files. This makes the target a clean,
learnable token stream instead of pixels or raw B-Rep.
"""

import random
import re

# ---------------------------------------------------------------------------
# Numeric quantization (mirrors the "geometry token generation" idea from
# Phase 1: continuous parameters -> discrete bins -> tokens)
# ---------------------------------------------------------------------------
NUM_BINS = [round(0.5 + 0.5 * i, 2) for i in range(20)]  # 0.5 .. 10.0 step 0.5


def value_to_bin(v):
    return min(range(len(NUM_BINS)), key=lambda i: abs(NUM_BINS[i] - v))


def sample_value():
    return random.choice(NUM_BINS)


# ---------------------------------------------------------------------------
# CAD token vocabulary (the "geometry/feature token" side of the tokenizer)
# ---------------------------------------------------------------------------
CAD_SPECIALS = ["<pad>", "<bos>", "<eos>"]
CAD_PRIMITIVES = ["BOX", "CYLINDER", "SPHERE", "SKETCH_RECT", "EXTRUDE"]
CAD_NUMS = [f"NUM_{i}" for i in range(len(NUM_BINS))]

CAD_VOCAB = CAD_SPECIALS + CAD_PRIMITIVES + CAD_NUMS
CAD_TOK2ID = {t: i for i, t in enumerate(CAD_VOCAB)}
CAD_ID2TOK = {i: t for t, i in CAD_TOK2ID.items()}

PAD_ID, BOS_ID, EOS_ID = CAD_TOK2ID["<pad>"], CAD_TOK2ID["<bos>"], CAD_TOK2ID["<eos>"]


# Token "type" ids used for the hierarchy/type embedding in the model:
# 0 = special, 1 = primitive/feature token, 2 = numeric parameter token
def cad_token_type(tok_id):
    tok = CAD_ID2TOK[tok_id]
    if tok in CAD_SPECIALS:
        return 0
    if tok in CAD_PRIMITIVES:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Shape generators: each returns (text, cad_token_list[no bos/eos])
# ---------------------------------------------------------------------------
BOX_TEMPLATES = [
    "Create a box that is {w} units wide, {h} units tall, and {d} units deep.",
    "Design a rectangular block with width {w}, height {h}, and depth {d}.",
    "I need a box sized {w} by {h} by {d}.",
    "Make a solid box, width {w}, height {h}, depth {d}.",
]
CYLINDER_TEMPLATES = [
    "Create a cylinder with radius {r} and height {h}.",
    "Make a cylindrical part {h} units tall with a radius of {r}.",
    "Design a cylinder, radius {r}, height {h}.",
]
SPHERE_TEMPLATES = [
    "Create a sphere with radius {r}.",
    "Design a ball of radius {r}.",
    "Make a spherical part with radius {r}.",
]
EXTRUDE_TEMPLATES = [
    "Sketch a rectangle {w} by {h} and extrude it {depth} units.",
    "Create an extruded block from a {w} by {h} rectangle, extruded {depth} units deep.",
    "Draw a {w} by {h} rectangle and extrude {depth} units.",
]


def gen_box():
    w, h, d = sample_value(), sample_value(), sample_value()
    text = random.choice(BOX_TEMPLATES).format(w=w, h=h, d=d)
    seq = ["BOX", f"NUM_{value_to_bin(w)}", f"NUM_{value_to_bin(h)}", f"NUM_{value_to_bin(d)}"]
    return text, seq


def gen_cylinder():
    r, h = sample_value(), sample_value()
    text = random.choice(CYLINDER_TEMPLATES).format(r=r, h=h)
    seq = ["CYLINDER", f"NUM_{value_to_bin(r)}", f"NUM_{value_to_bin(h)}"]
    return text, seq


def gen_sphere():
    r = sample_value()
    text = random.choice(SPHERE_TEMPLATES).format(r=r)
    seq = ["SPHERE", f"NUM_{value_to_bin(r)}"]
    return text, seq


def gen_extrude():
    w, h, depth = sample_value(), sample_value(), sample_value()
    text = random.choice(EXTRUDE_TEMPLATES).format(w=w, h=h, depth=depth)
    seq = [
        "SKETCH_RECT",
        f"NUM_{value_to_bin(w)}",
        f"NUM_{value_to_bin(h)}",
        "EXTRUDE",
        f"NUM_{value_to_bin(depth)}",
    ]
    return text, seq


GENERATORS = [gen_box, gen_cylinder, gen_sphere, gen_extrude]


def generate_example():
    text, seq = random.choice(GENERATORS)()
    cad_ids = [BOS_ID] + [CAD_TOK2ID[t] for t in seq] + [EOS_ID]
    return text, cad_ids


# ---------------------------------------------------------------------------
# Language tokenizer: small word-level vocab built from the synthetic corpus
# ---------------------------------------------------------------------------
_word_re = re.compile(r"[a-zA-Z]+|\d+\.\d+|\d+|[.,]")


def word_tokenize(text):
    return _word_re.findall(text.lower())


class LangTokenizer:
    def __init__(self):
        self.tok2id = {"<pad>": 0, "<unk>": 1}
        self.id2tok = {0: "<pad>", 1: "<unk>"}

    def build_vocab(self, texts):
        vocab = set()
        for t in texts:
            vocab.update(word_tokenize(t))
        for w in sorted(vocab):
            if w not in self.tok2id:
                idx = len(self.tok2id)
                self.tok2id[w] = idx
                self.id2tok[idx] = w

    def encode(self, text):
        return [self.tok2id.get(w, 1) for w in word_tokenize(text)]

    def __len__(self):
        return len(self.tok2id)


def build_dataset(n, lang_tok=None):
    """Generate n (text, cad_ids) pairs. If lang_tok has no vocab yet, build it."""
    raw = [generate_example() for _ in range(n)]
    texts = [t for t, _ in raw]
    if lang_tok is not None and len(lang_tok) <= 2:
        lang_tok.build_vocab(texts)
    return raw
