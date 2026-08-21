import sys
sys.path.insert(0, 'src')

from cadgensis.distillation.teachers.open_source_teacher import OpenSourceTeacher


def test_open_source_teacher_init():
    teacher = OpenSourceTeacher()
    assert teacher is not None