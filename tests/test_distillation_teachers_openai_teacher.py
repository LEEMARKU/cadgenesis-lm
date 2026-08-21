import sys
sys.path.insert(0, 'src')

from cadgensis.distillation.teachers.openai_teacher import OpenAITeacher


def test_openai_teacher_init():
    teacher = OpenAITeacher()
    assert teacher is not None