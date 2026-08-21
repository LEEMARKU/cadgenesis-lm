# CAD Intelligence — Documentation

Documentation for the `src/cadgenesis/cad/` package (Pillar 2 "CAD
Intelligence", CADGenesis-LM v6.0).

## Contents

| Document | Purpose |
| --- | --- |
| `architecture.md` | Package layout, module index, design principles |
| `uml_architecture.md` | Architecture & data-flow diagrams, subsystem relationships |
| `api_reference.md` | Concise, runnable usage reference for the main entry points |
| `development.md` | How to extend, test, lint, type-check and benchmark |

## Quick start

```bash
pip install -e .[dev]
python -m pytest tests/cad -q
python -m ruff check src/cadgenesis/cad tests/cad
python -m mypy --no-incremental src/cadgenesis/cad
python -m cadgenesis.cad.benchmarks.cad_benchmarks --reps 20
```

See `docs/v6_roadmap.md` for where this package fits in the overall v6.0 plan.