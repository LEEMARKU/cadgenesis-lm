# CAD Intelligence — Development Guide

How to extend, test, benchmark and type-check the `cadgenesis.cad` package.

## Environment

The CAD kernels are pure Python. The only runtime dependency of the whole
project is `torch`; dev tooling used in this package:

| Tool | Command |
| --- | --- |
| Tests | `python -m pytest tests/cad -q` |
| Lint | `python -m ruff check src/cadgenesis/cad tests/cad` |
| Type-check | `python -m mypy --no-incremental src/cadgenesis/cad` |
| Benchmarks | `python -m cadgenesis.cad.benchmarks.cad_benchmarks --reps 20` |

`ruff` rule set and line length come from `pyproject.toml`
(`select = E,F,I,UP,B,SIM,RUF,PERF,PYI`, `line-length = 100`). mypy targets
Python 3.10 via `[tool.mypy]`.

## Adding a new feature type

1. Subclass `Feature` in `cad/features/` and set the class attributes
   `type = FeatureType.<NEW>` and `operation`.
2. Decorate with `@register_feature` so the feature registry auto-discovers it.
3. Implement `validate()` returning a list of problems (empty == valid).
4. Add the enum member to `FeatureType` in `cad/features/base.py`.
5. Cover it in `tests/cad/test_features.py`.

## Adding a new validation check

1. Add a `check_*` function in `cad/validation/checks.py` returning a list of
   `CadCheckResult`.
2. Wire it into `CadValidator` in `cad/validation/pipeline.py` (respect the
   existing `analyze_*` / `check_*` boolean gates).
3. Export it in the module `__all__` and cover it in
   `tests/cad/test_validation.py`.

## Integration contract

- `integration/tokenizer_bridge.py` maps CAD objects to `CADTokenSequence`
  examples via the existing `AutonomousCADTokenizer` — do not change the
  tokenizer's public API, only the bridge.
- `integration/memory_bridge.py` adapts the `CADMemory` API — `store_design`,
  `store_brep`, `store_feature_tree` and `recall` are the public methods.
- `integration/reasoning_bridge.py` adapts geometry/constraint/DFM inputs to
  the reasoning toolkit's `Variable` / `Constraint` / part-dict formats.
- End-to-end coverage lives in `tests/cad/test_integration.py` (builds a real
  `AutonomousCADTokenizer.build_mini()`).

## Gotchas

- `Vec` subclasses `tuple`; overriding `__mul__`/`__rmul__` needs
  `# type: ignore[override]` because of the LSP check against `tuple`.
- All `to_dict()` / `from_dict()` round-trips must preserve equality of the
  re-parsed object (verified for assemblies, GD&T and materials).
- Mesh IO is symmetric: STL expands vertices per-face on import, so a round
  trip is vertex-count preserving for OBJ/PLY but not for STL.
- `FeatureTree` rejects duplicate names and `Sketch` rejects duplicate entity
  names; reuse a builder pattern instead of mutating names.
