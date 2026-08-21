"""
cadgenesis.train
================
Compatibility shim — the training CLI now lives in ``cadgenesis.cli.train``.
Kept so ``python -m cadgenesis.train`` and ``from cadgenesis.train import main``
continue to work unchanged.
"""

from cadgenesis.cli.train import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    main()
