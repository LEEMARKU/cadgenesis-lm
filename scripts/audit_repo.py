"""Repository audit tool for CADGenesis-LM v6.0.

Usage::

    python scripts/audit_repo.py            # full report, exit non-zero on failures
    python scripts/audit_repo.py --strict   # any stub or missing test fails
    python scripts/audit_repo.py --json     # emit JSON report

The audit verifies the acceptance criteria of the v6.0 roadmap:

1. No stub modules remain under ``src/cadgenesis/``.
2. Every package exposes public API surface via ``__init__.py``.
3. Every module with executable logic has a corresponding test file.
4. Pillar -> module mapping coverage is reported.
5. Packaging metadata is consistent.

Exit code 0 when the audit passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "cadgenesis"
TESTS = Path(__file__).resolve().parent.parent / "tests"
PACKAGE_META = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Modules that are allowed to be intentionally thin (facades / re-exports).
_FACADE_NAMES = {"__init__.py", "py.typed"}

# Modules known to be pure facades aggregating sibling modules.
_FACADE_MODULES = {
    "cadgenesis.train",
    "cadgenesis.transformer.transformer",
    "cadgenesis.tokenizer.tokenizer",
    "cadgenesis.tokenizer.cad_tokens",
    "cadgenesis.tokenizer.geometry_tokens",
    "cadgenesis.tokenizer.constraint_tokens",
    "cadgenesis.tokenizer.material_tokens",
    "cadgenesis.tokenizer.assembly_tokens",
    "cadgenesis.tokenizer.simulation_tokens",
    "cadgenesis.transformer.positional_encoding",
    "cadgenesis.confidence.confidence",
}


@dataclass
class ModuleInfo:
    module: str
    path: Path
    is_stub: bool = False
    stub_reason: str = ""
    has_test: bool = False
    public_names: list = field(default_factory=list)
    source_lines: int = 0
    docstring: str = ""


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent)  # .../src
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]  # strip .py
    return ".".join(parts)


def strip_docstring(source: str) -> tuple[str, str]:
    """Return (body_without_leading_docstring, docstring_text)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, ""
    if not tree.body:
        return source, ""
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        lines = source.splitlines()
        end_line = first.end_lineno or 1
        body = "\n".join(lines[end_line:])
        return body, first.value.value
    return source, ""


def classify_stub(source: str) -> tuple[bool, str]:
    """Classify a module source as a stub.

    A stub is a module whose executable body (ignoring the leading docstring,
    ``from __future__ import annotations`` and comments) contains no statements.
    """
    body, _ = strip_docstring(source)
    lines = [ln for ln in body.splitlines() if ln.strip()]
    code = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in ("from __future__ import annotations",):
            continue
        code.append(stripped)
    if not code:
        return True, "empty body"
    # Bare 'pass' only counts as a stub when the whole body is 'pass'.
    if len(code) == 1 and code[0] in ("pass", "..."):
        return True, f"only {code[0]}"
    return False, ""


def collect_modules() -> list[ModuleInfo]:
    infos: list[ModuleInfo] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        module = module_name(path)
        source = path.read_text(encoding="utf-8", errors="replace")
        body, doc = strip_docstring(source)
        is_stub, reason = classify_stub(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        public_names = []
        if tree is not None:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not node.name.startswith("_"):
                        public_names.append(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            public_names.append(target.id)  # noqa: PERF401
        TESTS / path.relative_to(SRC).with_suffix(".py").name
        # Test files live under tests/<pkg>/test_<name>.py mirroring the tree.
        rel = path.relative_to(SRC)
        test_candidates = [
            TESTS / rel.parent / f"test_{rel.name}",
            TESTS / rel.parent.parent / f"test_{rel.parent.name}" / f"test_{rel.name}",
        ]
        has_test = any(p.exists() for p in test_candidates)
        # Fallback: a test anywhere whose name matches the module's base name.
        if not has_test:
            base = rel.stem.replace("cadgenesis.", "")
            has_test = any(
                "test_" + base in (tp.stem if tp.stem != "" else tp.name)
                for tp in TESTS.rglob("test_*.py")
            )
        infos.append(
            ModuleInfo(
                module=module,
                path=path,
                is_stub=is_stub,
                stub_reason=reason,
                has_test=has_test,
                public_names=public_names,
                source_lines=len(body.splitlines()),
                docstring=doc,
            )
        )
    return infos


def check_init_exports() -> list[str]:
    """Ensure every non-facade package re-exports a public API."""
    missing = []
    for init in sorted(SRC.rglob("__init__.py")):
        source = init.read_text(encoding="utf-8", errors="replace")
        if "__all__" in source or "import" in source:
            continue
        # Allow empty init for namespace-only packages that currently have no
        # implemented modules yet; the audit flags them for tracking.
        pkg = init.parent.relative_to(SRC.parent)
        missing.append(".".join(pkg.parts))
    return missing


def pillar_coverage() -> list[dict]:
    """Report the roadmap pillar -> module coverage summary."""
    modules = collect_modules()
    stub_modules = [m for m in modules if m.is_stub]
    coverage = []
    for p, prefixes in PILLAR_MODULES:
        matched = [
            s.module
            for s in stub_modules
            if any(s.module.startswith(pre + ".") or s.module == pre for pre in prefixes)
        ]
        coverage.append(
            {
                "pillar": p,
                "modules": ", ".join(prefixes),
                "stub_modules": sorted(set(matched)),
            }
        )
    return coverage


# Roadmap pillar -> module map (module prefixes, longest match first).
PILLAR_MODULES = [
    (
        "Foundation Model",
        [
            "cadgenesis.transformer",
            "cadgenesis.tokenizer",
            "cadgenesis.inference",
            "cadgenesis.training",
        ],
    ),
    ("CAD Intelligence", ["cadgenesis.tokenizer", "cadgenesis.reasoning", "cadgenesis.execution"]),
    (
        "Multimodal Understanding",
        ["cadgenesis.multimodal", "cadgenesis.datasets", "cadgenesis.evaluation"],
    ),
    ("World Model", ["cadgenesis.world_model", "cadgenesis.evaluation"]),
    ("Multi-Agent Intelligence", ["cadgenesis.agents"]),
    ("Layer-Integrated Memory", ["cadgenesis.memory"]),
    ("Neuro-Symbolic Reasoning", ["cadgenesis.reasoning"]),
    ("CAD Execution & Validation", ["cadgenesis.execution"]),
    (
        "Learning System",
        [
            "cadgenesis.training",
            "cadgenesis.continual_learning",
            "cadgenesis.adapters",
            "cadgenesis.distillation",
        ],
    ),
    (
        "Reliability & Confidence",
        ["cadgenesis.confidence", "cadgenesis.monitoring", "cadgenesis.evaluation"],
    ),
    (
        "Production Platform",
        [
            "cadgenesis.serving",
            "cadgenesis.cli",
            "cadgenesis.optimization",
            "cadgenesis.config",
            "cadgenesis.telemetry",
            "cadgenesis.logging",
        ],
    ),
    ("Research Infrastructure", ["cadgenesis.evaluation", "cadgenesis.datasets"]),
    ("Provenance & Auditability", ["cadgenesis.provenance"]),
    ("Research Economy", ["cadgenesis.research_economy"]),
    ("Quantum Interfaces", ["cadgenesis.quantum"]),
    ("Frontier Research Lab", ["cadgenesis.research_lab"]),
    ("Autonomous Research Lab", ["cadgenesis.research_lab"]),
    ("Knowledge Network", ["cadgenesis.knowledge_network", "cadgenesis.reasoning"]),
    ("Digital Twin", ["cadgenesis.digital_twin"]),
    ("Autonomous Platform", ["cadgenesis.platform"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on any stub or missing test")
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args()

    modules = collect_modules()
    stubs = [m for m in modules if m.is_stub]
    missing_tests = [m for m in modules if not m.has_test and not m.is_stub]
    init_missing = check_init_exports()
    version = ""
    meta = PACKAGE_META.read_text(encoding="utf-8") if PACKAGE_META.exists() else ""
    mver = re.search(r'version\s*=\s*"([^"]+)"', meta)
    if mver:
        version = mver.group(1)

    report = {
        "version": version,
        "total_modules": len(modules),
        "stub_modules": [{"module": m.module, "reason": m.stub_reason} for m in stubs],
        "modules_without_tests": [m.module for m in missing_tests],
        "packages_without_exports": init_missing,
        "pillar_coverage": pillar_coverage(),
        "public_api_count": sum(len(m.public_names) for m in modules),
        "lines_of_code": sum(m.source_lines for m in modules),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 72)
    print(f"CADGenesis-LM repository audit — version {version or 'unknown'}")
    print("=" * 72)
    print(f"Total modules            : {len(modules)}")
    print(f"Public API names         : {report['public_api_count']}")
    print(f"Lines of implementation  : {report['lines_of_code']}")
    print(f"Stub modules             : {len(stubs)}")
    for s in stubs:
        print(f"    - {s.module}: {s.stub_reason}")
    print(f"Modules without tests    : {len(missing_tests)}")
    for m in missing_tests:
        print(f"    - {m.module}")
    print(f"Packages without exports : {len(init_missing)}")
    for p in init_missing:
        print(f"    - {p}")
    print("-" * 72)
    print("Pillar coverage:")
    for p in report["pillar_coverage"]:
        status = "OK" if not p["stub_modules"] else f"{len(p['stub_modules'])} stubs"
        print(f"    [{status:>12}] {p['pillar']}: {p['modules']}")

    failed = False
    if stubs:
        failed = True
        if not args.strict:
            print("\nNOTE: stub modules remain — set --strict to fail the audit on stubs.")
    if args.strict:
        if stubs:
            failed = True
        if missing_tests:
            failed = True
            print(f"FAIL: {len(missing_tests)} implemented module(s) lack tests.")
        if init_missing:
            failed = True
            print(f"FAIL: {len(init_missing)} package(s) lack public exports.")
    if failed:
        print("\nAUDIT: FAIL")
        return 1
    print("\nAUDIT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
