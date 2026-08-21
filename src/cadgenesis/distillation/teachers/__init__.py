"""Teacher model registry and factories."""

from cadgenesis.distillation.teachers.hf_teacher import HFTeacher
from cadgenesis.distillation.teachers.open_source_teacher import (
    MODEL_MAP,
    OpenSourceTeacher,
)
from cadgenesis.distillation.teachers.openai_teacher import OpenAITeacher

__all__ = ["MODEL_MAP", "HFTeacher", "OpenAITeacher", "OpenSourceTeacher"]
