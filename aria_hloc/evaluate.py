"""Evaluate localization results against an MPS trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aria.mps import Trajectory
from .geometry import relative_pose_error, transform_from_dict

DEFAULT_THRESHOLDS: Tuple[Tuple[float, float], ...] = ((0.05, 2.0), (0.25, 5.0), (1.0, 10.0))


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform ``T_dst_src`` (no scale) aligning ``src`` points onto ``dst`` (least squares)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(src) < 3:
        raise ValueError("Need at least 3 correspondences for a rigid alignment")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class EvaluationSummary:
    num_queries: int
    num_localized: int
    num_with_ground_truth: int
    recall: Dict[str, float] = field(default_factory=dict)  # "0.25m/5deg" -> fraction of ALL queries
    median_translation_error_m: Optional[float] = None
    median_rotation_error_deg: Optional[float] = None
    mean_translation_error_m: Optional[float] = None
    per_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alignment: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _errors_for(frames: Sequence[Dict[str, Any]], trajectory: Trajectory, T_align: Optional[np.ndarray]):
    errors = []  # (frame, dt, dR)
    num_gt = 0
    for fr in frames:
        gt = trajectory.pose_at(int(fr["timestamp_ns"]))
        if gt is None:
            continue
        num_gt += 1
        if not fr.get("success") or fr.get("T_world_device") is None:
            errors.append((fr, None, None))
            continue
        T_est = transform_from_dict(fr["T_world_device"])
        if T_align is not None:
            T_est = T_align @ T_est
        dt, dR = relative_pose_error(T_est, gt.T_world_device)
        errors.append((fr, dt, dR))
    return errors, num_gt


def evaluate_frames(
    frames: Sequence[Dict[str, Any]],
    trajectory: Trajectory,
    thresholds: Sequence[Tuple[float, float]] = DEFAULT_THRESHOLDS,
    align: bool = False,
) -> EvaluationSummary:
    """Compare localized device poses with the MPS ground-truth trajectory.

    Args:
        frames: result dictionaries with ``timestamp_ns``, ``label``, ``success``
            and ``T_world_device`` (see :mod:`aria_hloc.cli`).
        trajectory: ground-truth trajectory of the query recording.
        thresholds: ``(meters, degrees)`` pairs for recall.
        align: estimate a rigid alignment between the estimated and ground-truth
            positions first. Needed when the query recording was processed by
            MPS independently (its world frame differs from the map's).
    """
    T_align = None
    alignment = None
    if align:
        src, dst = [], []
        for fr in frames:
            if not fr.get("success") or fr.get("T_world_device") is None:
                continue
            gt = trajectory.pose_at(int(fr["timestamp_ns"]))
            if gt is None:
                continue
            src.append(transform_from_dict(fr["T_world_device"])[:3, 3])
            dst.append(gt.T_world_device[:3, 3])
        if len(src) >= 3:
            T_align = umeyama_se3(np.array(src), np.array(dst))
            # One robust re-fit: drop the worst 20% residuals.
            res = np.linalg.norm((np.array(src) @ T_align[:3, :3].T + T_align[:3, 3]) - np.array(dst), axis=1)
            keep = res <= np.percentile(res, 80)
            if keep.sum() >= 3:
                T_align = umeyama_se3(np.array(src)[keep], np.array(dst)[keep])
            alignment = {"T_gt_from_map": T_align.tolist(), "num_correspondences": int(len(src))}

    errors, num_gt = _errors_for(frames, trajectory, T_align)
    summary = EvaluationSummary(
        num_queries=len(frames),
        num_localized=int(sum(1 for fr in frames if fr.get("success"))),
        num_with_ground_truth=num_gt,
        alignment=alignment,
    )
    valid = [(dt, dR) for _, dt, dR in errors if dt is not None]
    if valid:
        dts = np.array([v[0] for v in valid])
        dRs = np.array([v[1] for v in valid])
        summary.median_translation_error_m = float(np.median(dts))
        summary.median_rotation_error_deg = float(np.median(dRs))
        summary.mean_translation_error_m = float(np.mean(dts))
    for tm, td in thresholds:
        ok = sum(1 for dt, dR in valid if dt <= tm and dR <= td)
        summary.recall[f"{tm:g}m/{td:g}deg"] = float(ok / max(num_gt, 1))
    labels = sorted({fr.get("label", "") for fr, _, _ in errors})
    for label in labels:
        sub = [(dt, dR) for fr, dt, dR in errors if fr.get("label", "") == label]
        vals = [(dt, dR) for dt, dR in sub if dt is not None]
        entry: Dict[str, Any] = {"num": len(sub), "num_localized": len(vals)}
        if vals:
            entry["median_translation_error_m"] = float(np.median([v[0] for v in vals]))
            entry["median_rotation_error_deg"] = float(np.median([v[1] for v in vals]))
        for tm, td in thresholds:
            ok = sum(1 for dt, dR in vals if dt <= tm and dR <= td)
            entry[f"recall@{tm:g}m/{td:g}deg"] = float(ok / max(len(sub), 1))
        summary.per_camera[label] = entry
    return summary


def attach_errors(frames: List[Dict[str, Any]], trajectory: Trajectory, T_align: Optional[np.ndarray] = None) -> None:
    """Add ``error_translation_m``/``error_rotation_deg`` to each frame dict in place."""
    for fr, dt, dR in _errors_for(frames, trajectory, T_align)[0]:
        fr["error_translation_m"] = dt
        fr["error_rotation_deg"] = dR
