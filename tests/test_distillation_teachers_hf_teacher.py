import sys
sys.path.insert(0, 'src')

from cadgensis.distillation.teachers.hf_teacher import HFTeacher


def test_hf_teacher_init():
    teacher = HFTeacher()
    assert teacher is not None