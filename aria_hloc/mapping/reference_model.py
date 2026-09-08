"""Write a COLMAP model whose image poses come from the MPS trajectory.

hloc's ``triangulation`` uses this "reference" model (cameras + posed images,
no 3D points) to triangulate the matched features, which yields an hloc map in
the MPS world frame with metric scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..aria.calibration import RectifiedCamera
from ..geometry import invert_transform, rotmat_to_quat_wxyz
from ..map_format import Keyframe
from ..utils.read_write_model import Camera, Image, write_model


def build_colmap_model(cameras: Dict[str, RectifiedCamera], keyframes: Sequence[Keyframe]):
    """Return ``(cameras, images, points3D)`` dictionaries in COLMAP format."""
    camera_ids = {label: i + 1 for i, label in enumerate(sorted(cameras))}
    colmap_cameras = {}
    for label, cam_id in camera_ids.items():
        p = cameras[label].pinhole
        colmap_cameras[cam_id] = Camera(
            id=cam_id, model="PINHOLE", width=int(p.width), height=int(p.height), params=np.array(p.colmap_params)
        )
    colmap_images = {}
    for i, kf in enumerate(keyframes):
        if kf.label not in camera_ids:
            raise KeyError(f"Keyframe {kf.name} uses unknown camera {kf.label}")
        T_cam_world = invert_transform(kf.T_world_camera)
        colmap_images[i + 1] = Image(
            id=i + 1,
            qvec=rotmat_to_quat_wxyz(T_cam_world[:3, :3]),
            tvec=T_cam_world[:3, 3].copy(),
            camera_id=camera_ids[kf.label],
            name=kf.name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
    return colmap_cameras, colmap_images, {}


def write_reference_model(
    cameras: Dict[str, RectifiedCamera], keyframes: Sequence[Keyframe], out_dir: Path, ext: str = ".bin"
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c, i, p = build_colmap_model(cameras, keyframes)
    write_model(c, i, p, str(out_dir), ext=ext)
    return out_dir
