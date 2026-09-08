import numpy as np
import pytest

from aria_hloc import geometry as G


def random_rotation(rng):
    q = rng.normal(size=4)
    return G.quat_wxyz_to_rotmat(q / np.linalg.norm(q))


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = random_rotation(rng)
        q = G.rotmat_to_quat_wxyz(R)
        assert q[0] >= 0
        np.testing.assert_allclose(G.quat_wxyz_to_rotmat(q), R, atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_invert_and_points():
    rng = np.random.default_rng(1)
    T = G.make_transform(random_rotation(rng), rng.normal(size=3))
    np.testing.assert_allclose(T @ G.invert_transform(T), np.eye(4), atol=1e-12)
    p = rng.normal(size=(5, 3))
    q = G.transform_points(T, p)
    np.testing.assert_allclose(G.transform_points(G.invert_transform(T), q), p, atol=1e-12)


def test_rotation_angle_and_pose_error():
    Rz = G.quat_wxyz_to_rotmat([np.cos(np.radians(15)), 0, 0, np.sin(np.radians(15))])
    assert G.rotation_angle_deg(Rz) == pytest.approx(30.0)
    T0 = np.eye(4)
    T1 = G.make_transform(Rz, [3.0, 4.0, 0.0])
    dt, dR = G.relative_pose_error(T1, T0)
    assert dt == pytest.approx(5.0)
    assert dR == pytest.approx(30.0)


def test_slerp_endpoints_and_midpoint():
    q0 = np.array([1.0, 0, 0, 0])
    q1 = np.array([np.cos(np.radians(45)), 0, 0, np.sin(np.radians(45))])  # 90 deg about z
    np.testing.assert_allclose(G.slerp_wxyz(q0, q1, 0.0), q0)
    np.testing.assert_allclose(G.slerp_wxyz(q0, q1, 1.0), q1)
    mid = G.slerp_wxyz(q0, q1, 0.5)
    assert G.rotation_angle_deg(G.quat_wxyz_to_rotmat(mid)) == pytest.approx(45.0)
    # Antipodal representation must give the same result.
    np.testing.assert_allclose(np.abs(G.slerp_wxyz(q0, -q1, 0.5)), np.abs(mid), atol=1e-12)


def test_interpolate_transform():
    T0 = np.eye(4)
    T1 = G.make_transform(G.quat_wxyz_to_rotmat([np.cos(np.radians(30)), 0, np.sin(np.radians(30)), 0]), [2, 0, 0])
    Tm = G.interpolate_transform(T0, T1, 0.25)
    np.testing.assert_allclose(Tm[:3, 3], [0.5, 0, 0])
    assert G.rotation_angle_deg(Tm[:3, :3]) == pytest.approx(15.0)


def test_transform_dict_roundtrip():
    rng = np.random.default_rng(2)
    T = G.make_transform(random_rotation(rng), rng.normal(size=3))
    d = G.transform_to_dict(T)
    np.testing.assert_allclose(G.transform_from_dict(d), T, atol=1e-12)
    np.testing.assert_allclose(G.transform_from_dict({k: v for k, v in d.items() if k != "matrix"}), T, atol=1e-10)
