import gzip
from pathlib import Path

import numpy as np
import pytest

from aria_hloc.aria import mps
from aria_hloc.geometry import PoseStamped, quat_wxyz_to_rotmat, rotmat_to_quat_wxyz, transform_from_quat_wxyz

HEADER = (
    "graph_uid,tracking_timestamp_us,utc_timestamp_ns,tx_world_device,ty_world_device,tz_world_device,"
    "qx_world_device,qy_world_device,qz_world_device,qw_world_device,device_linear_velocity_x_device,"
    "device_linear_velocity_y_device,device_linear_velocity_z_device,angular_velocity_x_device,"
    "angular_velocity_y_device,angular_velocity_z_device,gravity_x_world,gravity_y_world,gravity_z_world,quality_score\n"
)


def write_trajectory(path: Path, n=11, dt_us=1000, uid="g1"):
    rows = [HEADER]
    for i in range(n):
        angle = np.radians(10 * i)
        q = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]  # wxyz, rotation about z
        rows.append(
            f"{uid},{i * dt_us},0,{0.1 * i},{0.2 * i},1.5,{q[1]},{q[2]},{q[3]},{q[0]},"
            "0,0,0,0,0,0,0,0,-9.81,1.0\n"
        )
    path.write_text("".join(rows))


def test_read_and_interpolate(tmp_path):
    p = tmp_path / "closed_loop_trajectory.csv"
    write_trajectory(p)
    traj = mps.read_trajectory_csv(p)
    assert len(traj) == 11
    assert traj.graph_uids() == ["g1"]
    assert traj.start_ns == 0 and traj.end_ns == 10 * 1000 * 1000
    # exact sample
    pose = traj.pose_at(3 * 1_000_000)
    np.testing.assert_allclose(pose.T_world_device[:3, 3], [0.3, 0.6, 1.5])
    # midpoint interpolation: translation linear, rotation half way
    pose = traj.pose_at(3_500_000)
    np.testing.assert_allclose(pose.T_world_device[:3, 3], [0.35, 0.7, 1.5], atol=1e-9)
    q = rotmat_to_quat_wxyz(pose.T_world_device[:3, :3])
    angle = 2 * np.degrees(np.arctan2(q[3], q[0]))
    assert angle == pytest.approx(35.0, abs=1e-6)
    assert pose.quality_score == 1.0
    # slightly outside the trajectory: snapped to the closest sample
    assert traj.pose_at(-10_000_000) is not None
    # far outside the trajectory
    assert traj.pose_at(-100_000_000) is None
    assert traj.pose_at(200_000_000) is None
    # nearest instead of interpolation
    pose = traj.pose_at(3_400_000, interpolate=False)
    np.testing.assert_allclose(pose.T_world_device[:3, 3], [0.3, 0.6, 1.5])


def test_gap_and_graph_boundaries(tmp_path):
    p = tmp_path / "closed_loop_trajectory.csv"
    write_trajectory(p, n=4, dt_us=1_000_000)  # 1 s apart
    traj = mps.read_trajectory_csv(p)
    assert traj.pose_at(1_500_000_000, max_gap_ns=100_000_000) is None
    assert traj.pose_at(1_500_000_000, max_gap_ns=2_000_000_000) is not None
    # different graphs must not be interpolated across
    poses = [
        PoseStamped(0, np.eye(4), "a"),
        PoseStamped(1000, transform_from_quat_wxyz([1, 0, 0, 0], [1, 0, 0]), "b"),
    ]
    assert mps.Trajectory(poses).pose_at(500, max_gap_ns=10_000) is None


def test_min_quality(tmp_path):
    p = tmp_path / "closed_loop_trajectory.csv"
    write_trajectory(p, n=3)
    traj = mps.read_trajectory_csv(p)
    assert traj.pose_at(500_000, min_quality=0.5) is not None
    assert traj.pose_at(500_000, min_quality=1.5) is None


def test_write_roundtrip(tmp_path):
    p = tmp_path / "closed_loop_trajectory.csv"
    write_trajectory(p, n=5)
    traj = mps.read_trajectory_csv(p)
    out = tmp_path / "out.csv"
    mps.write_trajectory_csv(traj, out)
    back = mps.read_trajectory_csv(out)
    for a, b in zip(traj, back):
        assert a.timestamp_ns == b.timestamp_ns
        np.testing.assert_allclose(a.T_world_device, b.T_world_device, atol=1e-8)


def test_find_mps_paths(tmp_path):
    root = tmp_path / "mps_rec_vrs"
    slam = root / "slam"
    slam.mkdir(parents=True)
    write_trajectory(slam / "closed_loop_trajectory.csv")
    (slam / "online_calibration.jsonl").write_text("")
    for q in (root, slam, slam / "closed_loop_trajectory.csv"):
        paths = mps.find_mps_paths(q)
        assert paths.closed_loop_trajectory == slam / "closed_loop_trajectory.csv"
        assert paths.online_calibration == slam / "online_calibration.jsonl"
        assert paths.semidense_points is None
    with pytest.raises(FileNotFoundError):
        mps.find_mps_paths(tmp_path / "nope")
    # ambiguous
    other = root / "slam2"
    other.mkdir()
    write_trajectory(other / "closed_loop_trajectory.csv")
    with pytest.raises(ValueError):
        mps.find_mps_paths(root)


def test_semidense_points(tmp_path):
    p = tmp_path / "semidense_points.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("uid,graph_uid,px_world,py_world,pz_world,inv_dist_std,dist_std,num_observations\n")
        f.write("1,g,0,0,1,0.001,0.01,5\n")
        f.write("2,g,1,0,1,0.1,0.5,2\n")
    pts = mps.read_semidense_points(p)
    assert pts.xyz.shape == (2, 3)
    good = mps.filter_semidense_points(pts)
    assert len(good.xyz) == 1 and good.uid[0] == 1
