import numpy as np

from aria_hloc.geometry import make_transform, quat_wxyz_to_rotmat
from aria_hloc.mapping.keyframes import select_keyframes


def test_translation_spacing():
    poses = [make_transform(np.eye(3), [0.05 * i, 0, 0]) for i in range(21)]  # 5 cm steps
    sel = select_keyframes(poses, min_translation_m=0.19, min_rotation_deg=360)
    assert sel == [0, 4, 8, 12, 16, 20]


def test_rotation_spacing_and_none():
    poses = []
    for i in range(10):
        a = np.radians(5 * i)
        poses.append(make_transform(quat_wxyz_to_rotmat([np.cos(a / 2), 0, 0, np.sin(a / 2)]), [0, 0, 0]))
    poses[3] = None
    sel = select_keyframes(poses, min_translation_m=10, min_rotation_deg=15)
    assert sel == [0, 4, 7]


def test_max_count():
    poses = [make_transform(np.eye(3), [i, 0, 0]) for i in range(100)]
    sel = select_keyframes(poses, min_translation_m=0.5, min_rotation_deg=10, max_count=10)
    assert len(sel) == 10 and sel[0] == 0 and sel[-1] == 99
