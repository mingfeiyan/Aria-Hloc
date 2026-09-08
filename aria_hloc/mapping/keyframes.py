"""Keyframe selection along a trajectory."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..geometry import rotation_angle_deg


def select_keyframes(
    poses: Sequence[Optional[np.ndarray]],
    min_translation_m: float = 0.15,
    min_rotation_deg: float = 10.0,
    max_count: Optional[int] = None,
) -> List[int]:
    """Greedy keyframe selection.

    Args:
        poses: per-frame ``T_world_device`` (4x4) or ``None`` when no pose is
            available (such frames are never selected).
        min_translation_m: keep a frame when it moved at least this far from
            the last keyframe.
        min_rotation_deg: or rotated at least this much.
        max_count: optional cap; the selection is thinned uniformly.

    Returns:
        Sorted list of selected frame indices.
    """
    selected: List[int] = []
    last: Optional[np.ndarray] = None
    for i, T in enumerate(poses):
        if T is None:
            continue
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        if last is None:
            selected.append(i)
            last = T
            continue
        dt = float(np.linalg.norm(T[:3, 3] - last[:3, 3]))
        dR = rotation_angle_deg(last[:3, :3].T @ T[:3, :3])
        if dt >= min_translation_m or dR >= min_rotation_deg:
            selected.append(i)
            last = T
    if max_count is not None and len(selected) > max_count > 0:
        idx = np.linspace(0, len(selected) - 1, max_count).round().astype(int)
        selected = sorted(set(selected[k] for k in idx))
    return selected
