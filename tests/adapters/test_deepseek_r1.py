"""
Tests for the DeepSeek-R1 teacher integration (no network / 3GB download needed;
the HF model is replaced by a fake model injected into the reasoner).
"""

import pytest
import torch

from cadgenesis.adapters.deepseek_r1 import (
    DeepSeekR1DataGenerator,
    DeepSeekR1Reasoner,
    DeepSeekR1Teacher,
)
from cadgenesis.tokenizer import AutonomousCADTokenizer

FAKE_TOK2ID = {
    "SKETCH_RECT": 0,
    "EXTRUDE": 1,
    "BOX": 2,
    "SKETCH_CIRCLE": 3,
    "CYLINDER": 4,
    "OTHER_THING": 5,
}
FAKE_ID2TOK = {v: k for k, v in FAKE_TOK2ID.items()}


class FakeTokenizer:
    pad_token = "<eos>"
    eos_token = "<eos>"
    pad_token_id = 3
    eos_token_id = 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "[SYSTEM]\n" + "\n".join(m["content"] for m in messages if m["role"] == "user")

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[3, 3, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return ", ".join(FAKE_ID2TOK.get(int(i), "") for i in ids if int(i) in FAKE_ID2TOK)


class FakeQwenModel:
    def __init__(self):
        self.generated_ids = [1, 0, 2]  # EXTRUDE, SKETCH_RECT, BOX

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        fake = torch.tensor([self.generated_ids])
        return torch.cat([input_ids, fake], dim=-1)

    def __call__(self, **kwargs):
        hidden = torch.randn(kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1], 4)
        return type("Out", (), {"hidden_states": (hidden,) * 3})()


@pytest.fixture()
def reasoner():
    return DeepSeekR1Reasoner(model=FakeQwenModel(), tokenizer=FakeTokenizer())


@pytest.fixture()
def mini_vocab():
    return AutonomousCADTokenizer.build_mini().vocab


def test_reasoner_is_loaded_when_injected(reasoner):
    assert reasoner.loaded


def test_reasoner_generate_calls_model_once(reasoner):
    out = reasoner.generate("design a box")
    assert out == "EXTRUDE, SKETCH_RECT, BOX"
    assert reasoner.model is not None


def test_reasoner_generate_reasoning(reasoner):
    out = reasoner.generate_reasoning("design a bracket")
    assert isinstance(out, str) and out


def test_reasoner_last_hidden_state_shape(reasoner):
    h = reasoner.last_hidden_state("design a box")
    assert tuple(h.shape) == (1, 3, 4)


def test_teacher_parse_feature_tokens_maps_known_tokens(reasoner, mini_vocab):
    teacher = DeepSeekR1Teacher(reasoner)
    program = teacher.parse_feature_tokens(
        "SKETCH_RECT then EXTRUDE, BOX; NOT_A_FEATURE", mini_vocab
    )
    id2tok = mini_vocab.to_id2tok()
    assert program and all(i > 0 for i in program)
    assert {id2tok[i] for i in program} == {"SKETCH_RECT", "EXTRUDE", "BOX"}


def test_teacher_parse_skips_unknown_tokens(reasoner, mini_vocab):
    teacher = DeepSeekR1Teacher(reasoner)
    program = teacher.parse_feature_tokens("BOGUS_FEATURE, ALSO_UNKNOWN", mini_vocab)
    assert program == []


def test_teacher_generate_cad_program(reasoner, mini_vocab):
    teacher = DeepSeekR1Teacher(reasoner)
    program, spec = teacher.generate_cad_program("design a box", mini_vocab)
    assert program  # EXTRUDE, SKETCH_RECT, BOX are all in the mini vocab
    assert spec == "EXTRUDE, SKETCH_RECT, BOX"


def test_teacher_implements_generate_cad_toon(reasoner):
    teacher = DeepSeekR1Teacher(reasoner)
    assert teacher.generate_cad_toon("design a box") == teacher.generate_cad_spec("design a box")


def test_data_generator_valid_records(reasoner, mini_vocab):
    gen = DeepSeekR1DataGenerator(DeepSeekR1Teacher(reasoner), vocab=mini_vocab)
    records = gen.generate_dataset(["make a box", "make a cylinder"])
    assert len(records) == 2
    for r in records:
        assert r["prompt"]
        assert r["spec_text"]
        assert r["program_ids"]
        assert r["valid"] is True
        assert "reasoning" in r


def test_data_generator_honors_validator(reasoner, mini_vocab):
    def validator(program_ids):
        return len(program_ids) > 2

    gen = DeepSeekR1DataGenerator(
        DeepSeekR1Teacher(reasoner), vocab=mini_vocab, validator=validator
    )
    records = gen.generate_dataset(["make a box"])
    assert records[0]["valid"] is (len(records[0]["program_ids"]) > 2)
