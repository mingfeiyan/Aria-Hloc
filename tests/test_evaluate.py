import numpy as np
import pytest

from aria_hloc.aria.mps import Trajectory
from aria_hloc.evaluate import evaluate_frames, umeyama_se3
from aria_hloc.geometry import PoseStamped, make_transform, quat_wxyz_to_rotmat, transform_to_dict


def test_umeyama():
    rng = np.random.default_rng(0)
    R = quat_wxyz_to_rotmat(rng.normal(size=4))
    t = rng.normal(size=3)
    src = rng.normal(size=(20, 3))
    dst = src @ R.T + t
    T = umeyama_se3(src, dst)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-10)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-10)


def make_trajectory(n=20):
    poses = []
    for i in range(n):
        a = np.radians(3 * i)
        T = make_transform(quat_wxyz_to_rotmat([np.cos(a / 2), 0, np.sin(a / 2), 0]), [0.1 * i, 0, 1])
        poses.append(PoseStamped(i * 100_000_000, T, "g"))
    return Trajectory(poses)


def test_evaluate_frames_recall():
    traj = make_trajectory()
    frames = []
    for i, p in enumerate(traj):
        T = p.T_world_device.copy()
        if i % 4 == 0:
            T[:3, 3] += [0.5, 0, 0]  # bad
        frames.append({"label": "camera-rgb", "timestamp_ns": p.timestamp_ns, "success": True, "T_world_device": transform_to_dict(T)})
    frames.append({"label": "camera-rgb", "timestamp_ns": 5 * 100_000_000 + 1, "success": False, "T_world_device": None})
    s = evaluate_frames(frames, traj)
    assert s.num_queries == 21 and s.num_localized == 20 and s.num_with_ground_truth == 21
    assert s.recall["0.25m/5deg"] == pytest.approx(15 / 21)
    assert s.recall["1m/10deg"] == pytest.approx(20 / 21)
    assert s.median_translation_error_m == pytest.approx(0.0, abs=1e-9)
    assert s.per_camera["camera-rgb"]["num_localized"] == 20


def test_evaluate_with_alignment():
    traj = make_trajectory()
    R = quat_wxyz_to_rotmat([np.cos(0.4), 0, 0, np.sin(0.4)])
    T_map_gt = make_transform(R, [5, -3, 0.2])  # map frame differs from GT frame
    frames = []
    for p in traj:
        T_est = np.linalg.inv(T_map_gt) @ p.T_world_device
        frames.append({"label": "camera-rgb", "timestamp_ns": p.timestamp_ns, "success": True, "T_world_device": transform_to_dict(T_est)})
    s = evaluate_frames(frames, traj, align=False)
    assert s.recall["0.05m/2deg"] == 0.0
    s = evaluate_frames(frames, traj, align=True)
    assert s.recall["0.05m/2deg"] == 1.0
    np.testing.assert_allclose(np.asarray(s.alignment["T_gt_from_map"]), T_map_gt, atol=1e-8)
