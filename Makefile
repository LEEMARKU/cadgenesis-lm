# CADGenesis-LM Makefile (Windows: run `make` via mingw32-make or use the
# equivalent commands directly).

.PHONY: install install-dev test bench clean lint format

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,bpe]"

test:
	$(PYTHON) -m pytest -q

bench:
	$(PYTHON) benchmarks/attention_benchmarks.py
	$(PYTHON) benchmarks/tokenizer_benchmarks.py

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in [pathlib.Path(x) for x in ['src/cadgenesis.egg-info', 'build', 'dist']] if p.exists()]"
	find . -type d -name __pycache__ -prune -exec rm -rf {} \; 2>/dev/null || true
	find . -type d -name .pytest_cache -prune -exec rm -rf {} \; 2>/dev/null || true
