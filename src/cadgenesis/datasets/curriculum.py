"""
cadgenesis.datasets.curriculum
==============================
Multi-category curriculum dataset for the CAD model (M3).

Builds on the validated template generator from :mod:`cad_program_synth` and
adds the production-curation stack the baseline audit found missing:

* **9 record categories** — ``nl2program``, ``nl2ir``, ``program2explanation``,
  ``geometry2description``, ``error2correction``, ``constraint``, ``parameter``,
  ``tool``, ``planning``;
* **quality filter** — syntax -> schema (CAD-IR) -> execute -> geometry ->
  constraint -> dedup -> score, producing a per-record quality score.
  Dedup is **exact** (content-hash program ID): dimension variants are kept
  on purpose, since they are the numeric-generalization curriculum.  The
  MinHash near-duplicate pass remains available at load time via
  :func:`cadgenesis.datasets.cad_jsonl.minhash_dedup`;
* **adversarial sets** — perturbed programs that the quality filter must reject;
* **leakage-free splits** — deterministic, type-stratified, disjoint by
  content-hash program ID;
* **JSONL + manifest output** with a content digest for reproducibility.

Record shape (superset of the legacy ``{"text", "cad"}``):

    {"text": str, "cad": [str, ...], "type": str,
     "program_id": str, "score": float, "quality": {...}}

:class:`CADJsonlDataset` ignores the extra keys, so curriculum files load with
the existing pipeline unchanged.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from cadgenesis.datasets.cad_program_synth import (
    _SLOT_KEYS,
    _TEMPLATES,
    _enhance_tokens,
    _num_token,
    _validate_tokens,
)
from cadgenesis.execution.freecad_engine import FreeCADEngine
from cadgenesis.ir import parse_program, validate_program_ir

#: All supported record categories (mission M3 list).
RECORD_TYPES: tuple[str, ...] = (
    "nl2program",
    "nl2ir",
    "program2explanation",
    "geometry2description",
    "error2correction",
    "constraint",
    "parameter",
    "tool",
    "planning",
)

#: Weight of each quality stage; total score is the weighted mean.
_QUALITY_WEIGHTS: dict[str, float] = {
    "syntax": 1.0,
    "schema": 1.0,
    "execute": 1.0,
    "geometry": 0.6,
    "constraint": 0.6,
    "unique": 0.6,
}

#: Minimum weighted score for a record to enter the dataset.
QUALITY_THRESHOLD = 0.85

#: Raw-sample inflation to compensate exact-dedup losses (single-slot
#: templates collapse under ``_enhance_tokens``).  Measured ~13% dupe rate.
_DEDUP_SLACK = 1.3

_PART_NAMES = (
    "steel box",
    "mounting bracket",
    "cylindrical housing",
    "base plate",
    "counterbore hole",
    "slot",
    "dowel joint",
    "external thread",
    "bolt hole",
    "shaft",
    "complete bracket",
    "complex fixture",
)


def _stable_id(record: dict[str, Any]) -> str:
    """Deterministic content ID of a token program (CAD-IR program_id)."""
    return parse_program(record["cad"]).program_id


def _describe_program(tokens: list[str]) -> str:
    """Deterministic plain-language description of a token program."""
    parts: list[str] = []
    for step in parse_program(tokens).steps:
        if step.kind.startswith("PRIM_"):
            dims = [f"{v} mm" for k, v in sorted(step.params.items()) if k.startswith("d")]
            suffix = f" ({', '.join(dims)})" if dims else ""
            parts.append(step.tokens[0].lower() + suffix)
        elif step.kind.startswith("FEAT_"):
            parts.append(step.tokens[0].lower())
        else:
            parts.extend(str(a).lower() for a in step.params.get("attr", []))
    return " ".join(parts) or "a part"


def _prompt_variants(kind: str, tokens: list[str], rng: random.Random) -> str:
    """Build the NL text for a record of category ``kind``."""
    description = _describe_program(tokens)
    if kind == "nl2program":
        return f"create {description}"
    if kind == "nl2ir":
        return f"produce the structured program for: {description}"
    if kind == "program2explanation":
        return f"explain what this program builds: {' '.join(tokens)}"
    if kind == "geometry2description":
        return f"describe the geometry of: {' '.join(tokens)}"
    if kind == "error2correction":
        bad = _perturb(tokens, rng)
        return f"fix this program: {' '.join(bad)}"
    if kind == "constraint":
        return f"respecting the dimension constraints, create {description}"
    if kind == "parameter":
        dims = [
            round(v, 1)
            for s in parse_program(tokens).steps
            for v in s.params.values()
            if isinstance(v, (int, float))
        ]
        dim_txt = f" with parameters {dims}" if dims else ""
        return f"a parametrized part{dim_txt}: {description}"
    if kind == "tool":
        tool = rng.choice(("extrude", "hole", "fillet", "thread", "slot"))
        return f"use the {tool} tool to build {description}"
    if kind == "planning":
        return f"plan then execute: {description}"
    raise ValueError(f"unknown record type {kind}")


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def quality_filter(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the stage pipeline over records; return ``(kept, rejected)``.

    Stages (mission M3): syntax -> schema -> execute -> geometry -> constraint
    -> dedup -> score.  Every stage is measured per record into ``quality``.
    """
    from cadgenesis.execution.execution_engine import CADExecutionEngine
    from cadgenesis.execution.geometry_validation import validate_program

    engine = CADExecutionEngine()
    freecad = FreeCADEngine()
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for record in records:
        tokens = record["cad"]
        stages: dict[str, bool] = {}

        stages["syntax"] = bool(tokens) and validate_program(tokens)

        ir = parse_program(tokens)
        schema_report = validate_program_ir(ir, original=tokens)
        stages["schema"] = schema_report.passed

        result = engine.execute_and_evaluate(tokens)
        stages["execute"] = result.is_valid_geometry and not result.errors

        volume = float(freecad.execute(tokens).get("volume_mm3", 0.0))
        has_solids = bool(freecad.execute(tokens).get("solids"))
        stages["geometry"] = volume > 0.0 or not has_solids

        stages["constraint"] = all(
            isinstance(v, (int, float)) and 0.0 <= v <= 1_000.0
            for s in ir.steps
            for v in s.params.values()
            if isinstance(v, (int, float))
        )

        program_id = ir.program_id
        stages["unique"] = program_id not in seen

        score = sum(
            stages.get(name, False) * weight for name, weight in _QUALITY_WEIGHTS.items()
        ) / sum(_QUALITY_WEIGHTS.values())
        record = {**record, "program_id": program_id, "score": round(score, 4), "quality": stages}
        if stages["unique"]:
            seen.add(program_id)
        if score >= QUALITY_THRESHOLD and stages["unique"]:
            kept.append(record)
        else:
            rejected.append(record)
    return kept, rejected


# ---------------------------------------------------------------------------
# Adversarial sets
# ---------------------------------------------------------------------------

_PERTURBATIONS = (
    "drop_base",
    "all_attributes",
    "empty",
    "unknown_tokens",
    "numeric_only",
)


def _perturb(tokens: list[str], rng: random.Random) -> list[str]:
    """Deterministically produce a broken variant of ``tokens``."""
    from cadgenesis.ir import is_base_token

    # Mirror the legacy gate's base keywords: removing every one of these
    # guarantees the variant is invalid.
    legacy_base = {"EXTRUDE", "BOX", "CYLINDER", "SKETCH_RECT"}
    choice = rng.choice(_PERTURBATIONS)
    if choice == "drop_base":
        kept = [t for t in tokens if not is_base_token(t) and t not in legacy_base]
        return kept or ["NUM_10"]
    if choice == "all_attributes":
        attrs = [
            t
            for t in tokens
            if not is_base_token(t) and t not in legacy_base and not t.startswith("NUM_")
        ]
        return attrs[:3] or ["STEEL", "WEIGHT"]
    if choice == "empty":
        return []
    if choice == "unknown_tokens":
        return ["QUANTUM_WIDGET"]
    return ["NUM_10"]


def adversarial_records(base_records: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Broken program records the quality filter must reject."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for record in base_records:
        bad = _perturb(record["cad"], rng)
        out.append(
            {
                "text": f"fix this broken program: {' '.join(bad)}",
                "cad": bad,
                "type": "adversarial",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Generation, splits, output
# ---------------------------------------------------------------------------


def _sample_program_fine(rng: random.Random) -> tuple[str, list[str]]:
    """Sample a (prompt, tokens) pair with 1 mm dimension granularity.

    The legacy generator steps values in 5 mm increments (31 options per
    slot), which saturates quickly under exact dedup.  This curriculum-side
    sampler reuses the validated template set but fills slots at 1 mm steps
    (5..155 mm, 151 options per slot) — every value stays inside the raw-mm
    / quantizer-bin token conventions, so all tokens remain registered.
    """
    for _ in range(40):
        template = _TEMPLATES[rng.randrange(len(_TEMPLATES))]
        template_slot_keys = {t for t in template["tokens"] if t in _SLOT_KEYS}
        values: dict[str, int] = {}
        for key in template_slot_keys:
            values[_SLOT_KEYS[key]] = 5 + rng.randrange(151)  # 5..155 mm, 1 mm steps
        tokens = [
            _num_token(values[_SLOT_KEYS[t]]) if t in template_slot_keys else t
            for t in template["tokens"]
        ]
        tokens = _enhance_tokens(tokens)
        fmt_kwargs = {_SLOT_KEYS[k]: values[_SLOT_KEYS[k]] for k in template_slot_keys}
        try:
            prompt = template["prompt"].format(**fmt_kwargs)
        except KeyError:
            continue
        if _validate_tokens(tokens):
            return prompt, tokens
    return "a 10mm x 10mm steel box", [
        "SKETCH_RECT",
        _num_token(10),
        "EXTRUDE",
        _num_token(10),
        "BOX",
    ]


def build_curriculum_records(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Build up to ``n`` quality-filtered curriculum records (deterministic).

    Categories are sampled round-robin with a per-type random stream (so
    different categories draw different programs).  Single-slot templates
    deduplicate under ``_enhance_tokens``, so the raw sample budget is
    inflated by ``_DEDUP_SLACK`` to yield ``n`` *unique* records after the
    quality filter's exact-dedup stage.
    """
    records: list[dict[str, Any]] = []
    raw_target = int(n * _DEDUP_SLACK)
    target_per_type = max(1, raw_target // len(RECORD_TYPES))
    for type_index, kind in enumerate(RECORD_TYPES):
        type_rng = random.Random(seed + 1 + type_index)
        for _ in range(target_per_type * 16):  # attempt budget per type
            if sum(r["type"] == kind for r in records) >= target_per_type:
                break
            prompt, tokens = _sample_program_fine(type_rng)
            prompt = _prompt_variants(kind, tokens, type_rng)
            records.append({"text": prompt, "cad": tokens, "type": kind})
            if len(records) >= raw_target:
                break
        if len(records) >= raw_target:
            break
    kept, _rejected = quality_filter(records)
    return _truncate_balanced(kept, n)


def _truncate_balanced(records: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Keep ``n`` records with equal share per category (round-robin)."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record.get("type", "other"), []).append(record)
    out: list[dict[str, Any]] = []
    cursors = {kind: 0 for kind in by_type}
    kinds = sorted(by_type)
    while len(out) < n and any(cursors[k] < len(by_type[k]) for k in kinds):
        for kind in kinds:
            if len(out) >= n:
                break
            cursor = cursors[kind]
            if cursor < len(by_type[kind]):
                out.append(by_type[kind][cursor])
                cursors[kind] = cursor + 1
    return out


def make_splits(
    records: list[dict[str, Any]],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic, type-stratified, leakage-free train/val/test split.

    Split is performed per type (so every category appears in every split),
    and program IDs never cross splits.
    """
    if not (0.0 < train_fraction < 1.0 and 0.0 <= val_fraction < 1.0):
        raise ValueError("invalid split fractions")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train + val must be < 1.0")
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record.get("type", "other"), []).append(record)
    for _kind, group in sorted(by_type.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_train = round(len(shuffled) * train_fraction)
        n_val = round(len(shuffled) * val_fraction)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [{"text": r["text"], "cad": r["cad"], "type": r.get("type")} for r in records],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_curriculum_jsonl(
    directory: str | Path,
    n: int,
    seed: int = 0,
    progress: bool = True,
) -> dict[str, Any]:
    """Write train/val/test JSONL files + a manifest into ``directory``."""
    records = build_curriculum_records(n, seed=seed)
    train, val, test = make_splits(records, seed=seed)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    def _write(name: str, group: list[dict[str, Any]]) -> None:
        with (target / name).open("w", encoding="utf-8") as fh:
            for record in group:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write("train.jsonl", train)
    _write("val.jsonl", val)
    _write("test.jsonl", test)

    manifest = {
        "schema_version": "1.0.0",
        "seed": seed,
        "requested": n,
        "train": {"count": len(train), "digest": _digest(train)},
        "val": {"count": len(val), "digest": _digest(val)},
        "test": {"count": len(test), "digest": _digest(test)},
        "types": {kind: sum(1 for r in records if r.get("type") == kind) for kind in RECORD_TYPES},
    }
    with (target / "dataset_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    if progress:
        print(f"[curriculum] {len(train)}/{len(val)}/{len(test)} (train/val/test) -> {target}")
    return manifest


__all__ = [
    "QUALITY_THRESHOLD",
    "RECORD_TYPES",
    "adversarial_records",
    "build_curriculum_records",
    "make_splits",
    "quality_filter",
    "write_curriculum_jsonl",
]
