"""
Inference + a minimal, real "CAD execution" step.

generate(): greedy autoregressive decoding from a text prompt -> CAD tokens.
parse_cad_sequence(): turns the token sequence back into structured geometry
    (this is the scaled-down stand-in for the Geometry Kernel / Topology
    Analysis / Geometry Validation stages from the Phase 1 CAD execution
    flow -- it's a real parser+validator, just against a toy primitive set
    instead of OpenCascade/FreeCAD).
render(): draws the resulting primitive with matplotlib, so you can visually
    confirm the model produced something geometrically sensible.
"""

import torch

from data import (  # type: ignore[attr-defined]  # repo-root data/ namespace shadows sibling module
    BOS_ID,
    CAD_ID2TOK,
    EOS_ID,
    NUM_BINS,
    cad_token_type,
)


@torch.no_grad()
def generate(model, lang_tok, text, device, max_len=16):
    model.eval()
    src_ids = lang_tok.encode(text)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = src == 0

    tgt = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_type = torch.tensor([[cad_token_type(t.item()) for t in tgt[0]]], device=device)
        logits = model(src, tgt, tgt_type, src_key_padding_mask=src_pad_mask)
        next_id = logits[0, -1].argmax(-1).item()
        tgt = torch.cat([tgt, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == EOS_ID:
            break
    return [CAD_ID2TOK[i.item()] for i in tgt[0]]


def parse_cad_sequence(tokens):
    """
    Validate + interpret a generated token sequence. Returns
    (is_valid, shape_dict_or_error_message) -- this is the real, working
    equivalent of the "Geometry Validation" gate in the Phase 1 CAD
    execution flow: structurally invalid sequences are rejected here, not
    silently accepted.
    """
    toks = [t for t in tokens if t not in ("<bos>", "<eos>", "<pad>")]

    def num(tok):
        if not tok.startswith("NUM_"):
            return None
        idx = int(tok.split("_")[1])
        if idx >= len(NUM_BINS):
            return None
        return NUM_BINS[idx]

    if not toks:
        return False, "empty sequence"

    head = toks[0]
    if head == "BOX" and len(toks) >= 4:
        w, h, d = num(toks[1]), num(toks[2]), num(toks[3])
        if None in (w, h, d):
            return False, "malformed BOX parameters"
        return True, {"type": "box", "w": w, "h": h, "d": d}

    if head == "CYLINDER" and len(toks) >= 3:
        r, h = num(toks[1]), num(toks[2])
        if None in (r, h):
            return False, "malformed CYLINDER parameters"
        return True, {"type": "cylinder", "r": r, "h": h}

    if head == "SPHERE" and len(toks) >= 2:
        r = num(toks[1])
        if r is None:
            return False, "malformed SPHERE parameters"
        return True, {"type": "sphere", "r": r}

    if head == "SKETCH_RECT" and len(toks) >= 5 and toks[3] == "EXTRUDE":
        w, h, depth = num(toks[1]), num(toks[2]), num(toks[4])
        if None in (w, h, depth):
            return False, "malformed EXTRUDE parameters"
        return True, {"type": "extrude", "w": w, "h": h, "depth": depth}

    return False, f"unrecognized/malformed sequence starting with {head}"


def render(shape, ax=None):
    """Draw the validated shape with matplotlib (box/cylinder/sphere/extrude)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection="3d")

    t = shape["type"]
    if t in ("box", "extrude"):
        w = shape.get("w", 1)
        h = shape.get("h", 1)
        d = shape.get("d", shape.get("depth", 1))
        x, y, z = np.indices((2, 2, 2)).astype(float)
        x, y, z = x * w, y * h, z * d
        ax.bar3d(0, 0, 0, w, h, d, alpha=0.6, shade=True)
    elif t == "cylinder":
        r, h = shape["r"], shape["h"]
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(0, h, 2)
        theta, z = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot_surface(x, y, z, alpha=0.6)
    elif t == "sphere":
        r = shape["r"]
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_surface(x, y, z, alpha=0.6)

    ax.set_title(str(shape))
    return ax
