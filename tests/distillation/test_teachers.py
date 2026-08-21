"""tests/distillation/test_teachers.py"""

from __future__ import annotations

import pytest

from cadgenesis.distillation.distill_pipeline import TeacherModelInterface
from cadgenesis.distillation.teachers.open_source_teacher import MODEL_MAP, OpenSourceTeacher
from cadgenesis.distillation.teachers.openai_teacher import OpenAITeacher
from sdk import toon_extended


def assert_parseable_toon(toon: str) -> None:
    objects = toon_extended.from_toon(toon)
    assert objects, "teacher output must be a parseable TOON payload"
    assert all(str(obj.get("feature", "")).strip() for obj in objects)


def test_openai_teacher_falls_back_without_api_key():
    teacher = OpenAITeacher(api_key=None)
    toon = teacher.generate_cad_toon("Design a mounting bracket.")
    assert_parseable_toon(toon)


def test_openai_teacher_falls_back_without_openai_package(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    teacher = OpenAITeacher()
    toon = teacher.generate_cad_toon("Design a mounting bracket.")
    assert_parseable_toon(toon)


def test_openai_teacher_default_model():
    assert OpenAITeacher().model == "gpt-4o"


def test_open_source_teacher_defaults_per_provider():
    assert MODEL_MAP == {
        "deepseek": "deepseek-chat",
        "qwen": "qwen2.5-72b-instruct",
        "claude": "claude-3-5-sonnet-20241022",
    }
    assert OpenSourceTeacher("deepseek").model == MODEL_MAP["deepseek"]
    assert OpenSourceTeacher("qwen").model == MODEL_MAP["qwen"]
    assert OpenSourceTeacher("claude").model == MODEL_MAP["claude"]


def test_open_source_teacher_falls_back_without_api_key():
    for provider in ("deepseek", "qwen", "claude"):
        teacher = OpenSourceTeacher(provider, api_key=None)
        toon = teacher.generate_cad_toon("Design a mounting bracket.")
        assert_parseable_toon(toon)


def test_open_source_teacher_falls_back_without_client_library(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    teacher = OpenSourceTeacher("deepseek")
    toon = teacher.generate_cad_toon("Design a mounting bracket.")
    assert_parseable_toon(toon)


def test_open_source_teacher_rejects_unknown_provider():
    with pytest.raises(ValueError):
        OpenSourceTeacher("gemini")


def test_fallback_outputs_pass_quality_filtering():
    for teacher in (
        OpenAITeacher(api_key=None),
        OpenSourceTeacher("deepseek", api_key=None),
        OpenSourceTeacher("qwen", api_key=None),
        OpenSourceTeacher("claude", api_key=None),
    ):
        toon = teacher.generate_cad_toon("Design a mounting bracket.")
        assert_parseable_toon(toon)


def test_subclasses_share_base_class_interface():
    assert issubclass(OpenAITeacher, TeacherModelInterface)
    assert issubclass(OpenSourceTeacher, TeacherModelInterface)
