# Contributing to CADGenesis-LM

Thank you for your interest in contributing to CADGenesis-LM! This document
outlines the conventions and workflows used in this repository.

## Repository layout

This repository follows the enterprise architecture described in `docs/architecture.md`:

- `src/cadgenesis/` — the package source (submodules for tokenizer, transformer,
  memory, agents, reasoning, execution, and more).
- `tests/` — pytest suite (mirrors the `cadgenesis` module layout).
- `benchmarks/` — performance benchmarks.
- `configs/`, `docs/`, `scripts/`, `tools/`, `deployments/`, `docker/`,
  `notebooks/`, `examples/`, `plugins/`, `sdk/`, `external/` — supporting areas.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev,bpe]"       # editable install + dev extras
python -m pytest -q               # run the full suite
```

## Development workflow

1. Create a branch from `main`.
2. Write code and tests; keep the module layout under `src/cadgenesis/`.
3. Run `python -m pytest -q` and `python benchmarks/attention_benchmarks.py`.
4. Submit a pull request with a clear description and changelog entry.

## Code style

- Follow the existing conventions in the package (docstrings, type hints,
  `from __future__ import annotations` where appropriate).
- Do not add comments unless they clarify non-obvious intent.

## Code of conduct

Contributions are governed by `CODE_OF_CONDUCT.md`.
