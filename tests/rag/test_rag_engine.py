"""Tests for the M6 RAG engine (semantic memory + generation augmentation)."""

from __future__ import annotations

from cadgenesis.ir import parse_program
from cadgenesis.rag.rag_engine import _REFERENCE_PREFIX, CADRAGEngine

_RECORDS = [
    {
        "text": "create a steel box (80 mm, 40 mm, 20 mm)",
        "cad": ["BOX", "NUM_80", "NUM_40", "NUM_20"],
    },
    {
        "text": "create a cylindrical housing with 60 mm radius and 30 mm height",
        "cad": ["CYLINDER", "NUM_60", "NUM_30"],
    },
    {"text": "create a sphere with 50 mm radius", "cad": ["SPHERE", "NUM_50"]},
]


def _indexed() -> CADRAGEngine:
    engine = CADRAGEngine(top_k=1)
    for record in _RECORDS:
        engine.index_record(record)
    return engine


def test_index_records_keys_by_program_id() -> None:
    engine = _indexed()
    assert len(engine.store) == 3
    for record in _RECORDS:
        program_id = parse_program(record["cad"]).program_id
        entry = engine.store.peek(program_id)
        assert entry is not None
        assert entry.content["cad"] == record["cad"]


def test_index_jsonl_counts_records(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(__import__("json").dumps(record) for record in _RECORDS) + "\n",
        encoding="utf-8",
    )
    engine = CADRAGEngine()
    assert engine.index_jsonl(path) == 3


def test_retrieve_finds_matching_record() -> None:
    engine = _indexed()
    result = engine.retrieve("create a steel box", top_k=1)
    assert result.top is not None
    assert result.top.entry.content["cad"][0] == "BOX"


def test_precision_at_k_hits_and_misses() -> None:
    engine = _indexed()
    box_id = parse_program(_RECORDS[0]["cad"]).program_id
    assert engine.precision_at_k("create a steel box", box_id, k=1) == 1.0
    assert engine.precision_at_k("create a sphere please", box_id, k=1) == 0.0


def test_template_precision_at_k_matches_operation_kinds() -> None:
    engine = _indexed()
    assert (
        engine.template_precision_at_k(
            "create a steel box", ["BOX", "NUM_80", "NUM_40", "NUM_20"], k=1
        )
        == 1.0
    )
    assert engine.template_precision_at_k("a sphere please", ["BOX"], k=1) == 0.0


def test_augmented_prompt_includes_reference() -> None:
    engine = _indexed()
    prompt = engine.augmented_prompt("create a steel box", top_k=1)
    assert prompt.startswith("create a steel box")
    assert _REFERENCE_PREFIX in prompt


def test_generate_returns_augmented_output() -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def greedy(self, text, max_len=24, **kwargs):
            self.calls.append(text)
            from cadgenesis.inference.engine import GenerationResult

            return GenerationResult(
                text=text,
                tokens=["BOX", "<eos>"],
                ids=[2, 3],
                confidence=0.5,
                per_token_confidence=[0.5, 0.5],
                toon="",
                stopped_on_eos=True,
            )

    engine = _indexed()
    fake = _FakeEngine()
    out = engine.generate(fake, "create a steel box", max_len=16)
    assert fake.calls and _REFERENCE_PREFIX in fake.calls[0]
    assert out["tokens"] == ["BOX", "<eos>"]
    assert out["stopped_on_eos"]
    assert out["elapsed_ms"] >= 0.0


def test_generate_uses_sampling_when_temperature_given() -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.sampled = False

        def sample(self, text, max_len=24, **kwargs):
            self.sampled = True
            from cadgenesis.inference.engine import GenerationResult

            return GenerationResult(
                text=text,
                tokens=["BOX", "<eos>"],
                ids=[2, 3],
                confidence=0.5,
                per_token_confidence=[0.5, 0.5],
                toon="",
                stopped_on_eos=True,
            )

    fake = _FakeEngine()
    _indexed().generate(fake, "create a steel box", temperature=0.8, top_k=30)
    assert fake.sampled


def test_benchmark_retrieval_aggregates() -> None:
    engine = _indexed()
    box_cad = _RECORDS[0]["cad"]
    sphere_cad = _RECORDS[2]["cad"]
    stats = engine.benchmark_retrieval(
        [("create a steel box", box_cad), ("create a sphere please", sphere_cad)], k=1
    )
    assert stats["n"] == 2
    assert stats["precision_at_k"] == 1.0
    assert stats["mean_ms"] >= 0.0


def test_benchmark_retrieval_exact_key_mode() -> None:
    engine = _indexed()
    box_id = parse_program(_RECORDS[0]["cad"]).program_id
    stats = engine.benchmark_retrieval([("create a steel box", box_id)], k=1, template=False)
    assert stats["precision_at_k"] == 1.0


def test_top_k_must_be_positive() -> None:
    try:
        CADRAGEngine(top_k=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for top_k=0")


def test_summary_reports_store() -> None:
    summary = _indexed().summary()
    assert summary["store"]["name"] == "cad-rag"
    assert summary["store"]["size"] == 3
    assert summary["top_k"] == 1
