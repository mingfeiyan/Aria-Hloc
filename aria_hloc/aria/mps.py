"""Readers for Aria Machine Perception Services (MPS) SLAM outputs.

The trajectory reader is pure Python (``csv`` + numpy) so that it works without
``projectaria_tools`` installed; online calibration and semi-dense points use
``projectaria_tools`` when available.

MPS SLAM output layout (per recording)::

    mps_<recording>_vrs/
      slam/
        closed_loop_trajectory.csv
        open_loop_trajectory.csv
        online_calibration.jsonl
        semidense_points.csv.gz
        semidense_observations.csv.gz
"""

from __future__ import annotations

import bisect
import csv
import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..geometry import PoseStamped, interpolate_transform, transform_from_quat_wxyz

logger = logging.getLogger(__name__)

CLOSED_LOOP_FILENAME = "closed_loop_trajectory.csv"
OPEN_LOOP_FILENAME = "open_loop_trajectory.csv"
ONLINE_CALIBRATION_FILENAME = "online_calibration.jsonl"
SEMIDENSE_POINTS_FILENAME = "semidense_points.csv.gz"
SEMIDENSE_OBSERVATIONS_FILENAME = "semidense_observations.csv.gz"


@dataclass
class MpsPaths:
    """Resolved paths of the MPS SLAM files we use."""

    closed_loop_trajectory: Path
    online_calibration: Optional[Path] = None
    semidense_points: Optional[Path] = None
    semidense_observations: Optional[Path] = None


def find_mps_paths(path: Path) -> MpsPaths:
    """Locate the MPS SLAM files below ``path``.

    ``path`` may point directly at ``closed_loop_trajectory.csv``, at the
    ``slam`` folder, or at the MPS output root (``mps_<name>_vrs``).
    """
    path = Path(path)
    if path.is_file():
        if path.name != CLOSED_LOOP_FILENAME:
            raise FileNotFoundError(
                f"Expected {CLOSED_LOOP_FILENAME} or a directory containing it, got {path}"
            )
        traj = path
    else:
        if not path.exists():
            raise FileNotFoundError(f"MPS path does not exist: {path}")
        candidates = sorted(path.rglob(CLOSED_LOOP_FILENAME))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No {CLOSED_LOOP_FILENAME} found below {path}")
        if len(candidates) > 1:
            listing = "\n  ".join(str(c) for c in candidates)
            raise ValueError(
                f"Found several {CLOSED_LOOP_FILENAME} files below {path}; "
                f"pass one explicitly:\n  {listing}"
            )
        traj = candidates[0]
    slam_dir = traj.parent

    def _opt(name: str) -> Optional[Path]:
        p = slam_dir / name
        return p if p.exists() else None

    return MpsPaths(
        closed_loop_trajectory=traj,
        online_calibration=_opt(ONLINE_CALIBRATION_FILENAME),
        semidense_points=_opt(SEMIDENSE_POINTS_FILENAME),
        semidense_observations=_opt(SEMIDENSE_OBSERVATIONS_FILENAME),
    )


@dataclass
class Trajectory:
    """A time-sorted device trajectory (``T_world_device`` per timestamp)."""

    poses: List[PoseStamped] = field(default_factory=list)

    def __post_init__(self):
        self.poses = sorted(self.poses, key=lambda p: p.timestamp_ns)
        self._timestamps = [p.timestamp_ns for p in self.poses]

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self) -> Iterable[PoseStamped]:
        return iter(self.poses)

    @property
    def timestamps_ns(self) -> np.ndarray:
        return np.asarray(self._timestamps, dtype=np.int64)

    @property
    def start_ns(self) -> int:
        return self._timestamps[0]

    @property
    def end_ns(self) -> int:
        return self._timestamps[-1]

    def graph_uids(self) -> List[str]:
        seen = []
        for p in self.poses:
            if p.graph_uid not in seen:
                seen.append(p.graph_uid)
        return seen

    def pose_at(
        self,
        timestamp_ns: int,
        max_gap_ns: int = 100_000_000,
        interpolate: bool = True,
        min_quality: Optional[float] = None,
    ) -> Optional[PoseStamped]:
        """Return the device pose at ``timestamp_ns`` (device time).

        The pose is interpolated between the two bracketing trajectory samples
        (SLERP for rotation, linear for translation). ``None`` is returned when
        the query is outside the trajectory, when the bracketing samples are
        further apart than ``max_gap_ns``, when they belong to different graphs,
        or when their quality is below ``min_quality``.
        """
        if len(self.poses) == 0:
            return None
        ts = self._timestamps
        i = bisect.bisect_left(ts, timestamp_ns)
        if i < len(ts) and ts[i] == timestamp_ns:
            p = self.poses[i]
            if min_quality is not None and p.quality_score is not None and p.quality_score < min_quality:
                return None
            return p
        if i == 0 or i == len(ts):
            # Outside the trajectory. Tolerate a small extrapolation gap by snapping.
            j = 0 if i == 0 else len(ts) - 1
            if abs(ts[j] - timestamp_ns) <= max_gap_ns // 2:
                p = self.poses[j]
                return PoseStamped(timestamp_ns, p.T_world_device.copy(), p.graph_uid, p.quality_score)
            return None
        p0, p1 = self.poses[i - 1], self.poses[i]
        if p1.timestamp_ns - p0.timestamp_ns > max_gap_ns:
            return None
        if p0.graph_uid != p1.graph_uid:
            return None
        if min_quality is not None:
            for p in (p0, p1):
                if p.quality_score is not None and p.quality_score < min_quality:
                    return None
        if not interpolate:
            nearest = p0 if timestamp_ns - p0.timestamp_ns <= p1.timestamp_ns - timestamp_ns else p1
            return PoseStamped(timestamp_ns, nearest.T_world_device.copy(), nearest.graph_uid, nearest.quality_score)
        alpha = (timestamp_ns - p0.timestamp_ns) / float(p1.timestamp_ns - p0.timestamp_ns)
        T = interpolate_transform(p0.T_world_device, p1.T_world_device, alpha)
        q = None
        if p0.quality_score is not None and p1.quality_score is not None:
            q = min(p0.quality_score, p1.quality_score)
        return PoseStamped(timestamp_ns, T, p0.graph_uid, q)


def _find_column(header: Sequence[str], *candidates: str) -> Optional[int]:
    for c in candidates:
        if c in header:
            return header.index(c)
    return None


def read_trajectory_csv(path: Path) -> Trajectory:
    """Parse an MPS ``closed_loop_trajectory.csv`` (or ``open_loop_trajectory.csv``).

    Closed-loop files store ``T_world_device`` as ``tx/ty/tz_world_device`` and
    ``qx/qy/qz/qw_world_device``; open-loop files use the ``odometry`` suffix.
    """
    path = Path(path)
    poses: List[PoseStamped] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        frame = "world" if "tx_world_device" in header else "odometry"
        i_ts = _find_column(header, "tracking_timestamp_us")
        if i_ts is None:
            raise ValueError(f"{path}: missing tracking_timestamp_us column")
        i_t = [header.index(f"t{a}_{frame}_device") for a in "xyz"]
        i_q = [header.index(f"q{a}_{frame}_device") for a in "wxyz"]
        i_uid = _find_column(header, "graph_uid", "session_uid")
        i_quality = _find_column(header, "quality_score")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts_ns = int(round(float(row[i_ts]) * 1000.0))
            t = [float(row[k]) for k in i_t]
            q = [float(row[k]) for k in i_q]
            uid = row[i_uid].strip() if i_uid is not None else ""
            quality = float(row[i_quality]) if i_quality is not None and row[i_quality] != "" else None
            poses.append(PoseStamped(ts_ns, transform_from_quat_wxyz(q, t), uid, quality))
    if not poses:
        raise ValueError(f"{path}: no poses found")
    traj = Trajectory(poses)
    logger.info(
        "Read %d poses from %s (%.1f s, graphs=%s)",
        len(traj),
        path.name,
        (traj.end_ns - traj.start_ns) / 1e9,
        traj.graph_uids(),
    )
    return traj


def write_trajectory_csv(traj: Trajectory, path: Path) -> None:
    """Write a trajectory in the MPS closed-loop CSV layout (subset of columns)."""
    from ..geometry import rotmat_to_quat_wxyz

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "graph_uid",
                "tracking_timestamp_us",
                "utc_timestamp_ns",
                "tx_world_device",
                "ty_world_device",
                "tz_world_device",
                "qx_world_device",
                "qy_world_device",
                "qz_world_device",
                "qw_world_device",
                "quality_score",
            ]
        )
        for p in traj:
            q = rotmat_to_quat_wxyz(p.T_world_device[:3, :3])
            t = p.T_world_device[:3, 3]
            w.writerow(
                [
                    p.graph_uid,
                    p.timestamp_ns // 1000,
                    0,
                    *[f"{v:.9f}" for v in t],
                    *[f"{v:.9f}" for v in (q[1], q[2], q[3], q[0])],
                    "" if p.quality_score is None else p.quality_score,
                ]
            )


@dataclass
class SemidensePoints:
    xyz: np.ndarray  # (N, 3) in world frame
    inv_dist_std: np.ndarray  # (N,)
    dist_std: np.ndarray  # (N,)
    num_observations: np.ndarray  # (N,)
    uid: np.ndarray  # (N,) int64


def read_semidense_points(path: Path) -> SemidensePoints:
    """Read ``semidense_points.csv.gz`` (pure Python)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    xyz, inv_std, dist_std, nobs, uids = [], [], [], [], []
    with opener(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        ix = [header.index(f"p{a}_world") for a in "xyz"]
        i_inv = _find_column(header, "inv_dist_std")
        i_dist = _find_column(header, "dist_std")
        i_n = _find_column(header, "num_observations")
        i_uid = _find_column(header, "uid")
        for row in reader:
            if not row:
                continue
            xyz.append([float(row[k]) for k in ix])
            inv_std.append(float(row[i_inv]) if i_inv is not None else np.nan)
            dist_std.append(float(row[i_dist]) if i_dist is not None else np.nan)
            nobs.append(int(row[i_n]) if i_n is not None else 0)
            uids.append(int(row[i_uid]) if i_uid is not None else len(uids))
    return SemidensePoints(
        np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(inv_std, dtype=np.float64),
        np.asarray(dist_std, dtype=np.float64),
        np.asarray(nobs, dtype=np.int64),
        np.asarray(uids, dtype=np.int64),
    )


def filter_semidense_points(
    points: SemidensePoints, max_inv_dist_std: float = 0.005, max_dist_std: float = 0.05
) -> SemidensePoints:
    """Keep confident points (thresholds follow the projectaria_tools defaults)."""
    keep = np.ones(len(points.xyz), dtype=bool)
    if np.isfinite(points.inv_dist_std).any():
        keep &= points.inv_dist_std <= max_inv_dist_std
    if np.isfinite(points.dist_std).any():
        keep &= points.dist_std <= max_dist_std
    return SemidensePoints(
        points.xyz[keep],
        points.inv_dist_std[keep],
        points.dist_std[keep],
        points.num_observations[keep],
        points.uid[keep],
    )


class OnlineCalibrationProvider:
    """Nearest-in-time lookup of the MPS online camera calibration.

    Requires ``projectaria_tools``. The online calibration refines the factory
    intrinsics/extrinsics per frame; using it for the mapping images slightly
    improves consistency with the MPS trajectory.
    """

    def __init__(self, path: Path):
        from projectaria_tools.core import mps as aria_mps

        self.path = Path(path)
        self._calibs = aria_mps.read_online_calibration(str(self.path))
        self._timestamps = [self._timestamp_ns(c) for c in self._calibs]
        order = np.argsort(self._timestamps)
        self._calibs = [self._calibs[i] for i in order]
        self._timestamps = [self._timestamps[i] for i in order]
        logger.info("Read %d online calibrations from %s", len(self._calibs), self.path.name)

    @staticmethod
    def _timestamp_ns(calib) -> int:
        td = calib.tracking_timestamp
        return int(round(td.total_seconds() * 1e9))

    def __len__(self) -> int:
        return len(self._calibs)

    def camera_calibration(self, label: str, timestamp_ns: int, max_gap_ns: int = 500_000_000):
        """Return the nearest online ``CameraCalibration`` for ``label`` or ``None``."""
        if not self._calibs:
            return None
        i = bisect.bisect_left(self._timestamps, timestamp_ns)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self._timestamps):
                gap = abs(self._timestamps[j] - timestamp_ns)
                if best is None or gap < best[0]:
                    best = (gap, j)
        if best is None or best[0] > max_gap_ns:
            return None
        calib = self._calibs[best[1]]
        try:
            return calib.get_camera_calib(label)
        except AttributeError:  # pragma: no cover - older projectaria_tools
            for cam in calib.camera_calibs:
                if cam.get_label() == label:
                    return cam
        return None
