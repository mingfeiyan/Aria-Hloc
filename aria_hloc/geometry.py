"""Numpy-only rigid transform helpers.

Conventions
-----------
* A pose is a 4x4 homogeneous matrix ``T_a_b`` mapping points from frame ``b``
  to frame ``a`` (``p_a = T_a_b @ p_b``). This matches the naming used by
  Project Aria (``T_world_device``) and by COLMAP (``cam_from_world``).
* Quaternions are stored as ``[w, x, y, z]`` unless explicitly stated. COLMAP
  text/binary models and Aria MPS CSVs use different orderings; the readers
  convert at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def quat_wxyz_to_rotmat(q) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat_wxyz(R) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a ``[w, x, y, z]`` quaternion (w >= 0)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def make_transform(R, t) -> np.ndarray:
    """Build a 4x4 transform from a rotation matrix and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def transform_from_quat_wxyz(q, t) -> np.ndarray:
    return make_transform(quat_wxyz_to_rotmat(q), t)


def invert_transform(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T, points) -> np.ndarray:
    """Apply ``T`` (4x4) to an ``(N, 3)`` array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R) -> float:
    """Angle (degrees) of a rotation matrix."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def relative_pose_error(T_est, T_gt) -> Tuple[float, float]:
    """Return ``(translation_error_m, rotation_error_deg)`` between two poses.

    Both poses must express the same frame relation (e.g. both ``T_world_cam``).
    """
    T_est = np.asarray(T_est, dtype=np.float64).reshape(4, 4)
    T_gt = np.asarray(T_gt, dtype=np.float64).reshape(4, 4)
    dt = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    dR = rotation_angle_deg(T_est[:3, :3].T @ T_gt[:3, :3])
    return dt, dR


def slerp_wxyz(q0, q1, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two ``[w, x, y, z]`` quaternions."""
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    theta = theta0 * alpha
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return s0 * q0 + s1 * q1


def interpolate_transform(T0, T1, alpha: float) -> np.ndarray:
    """Interpolate two poses: SLERP on rotation, LERP on translation."""
    T0 = np.asarray(T0, dtype=np.float64).reshape(4, 4)
    T1 = np.asarray(T1, dtype=np.float64).reshape(4, 4)
    q = slerp_wxyz(rotmat_to_quat_wxyz(T0[:3, :3]), rotmat_to_quat_wxyz(T1[:3, :3]), alpha)
    t = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    return transform_from_quat_wxyz(q, t)


def transform_to_dict(T) -> dict:
    """JSON friendly representation of a pose (matrix + quaternion + translation)."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    q = rotmat_to_quat_wxyz(T[:3, :3])
    return {
        "matrix": T.tolist(),
        "quaternion_wxyz": q.tolist(),
        "translation": T[:3, 3].tolist(),
    }


def transform_from_dict(d: dict) -> np.ndarray:
    if "matrix" in d:
        return np.asarray(d["matrix"], dtype=np.float64).reshape(4, 4)
    return transform_from_quat_wxyz(d["quaternion_wxyz"], d["translation"])


@dataclass
class PoseStamped:
    """A device pose with its timestamp (device time, nanoseconds)."""

    timestamp_ns: int
    T_world_device: np.ndarray
    graph_uid: str = ""
    quality_score: Optional[float] = None
