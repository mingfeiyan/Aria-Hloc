"""Build an hloc map from an Aria VRS recording and its MPS SLAM output.

Pipeline
--------
1. Read the MPS closed-loop trajectory (``T_world_device`` over time).
2. Walk the requested camera streams of the VRS, look up the device pose of
   every frame and select keyframes by motion (translation/rotation).
3. Rectify the keyframes to pinhole images (upright) and store them.
4. Write a COLMAP "reference" model with the keyframe poses
   (``T_world_camera = T_world_device * T_device_camera``).
5. Run hloc: global descriptors (retrieval), image pairs (from the known
   poses), local features, matching, and triangulation with the fixed poses.

The resulting map lives in the MPS world frame, is metric, and can be used
directly by :class:`aria_hloc.localization.Relocalizer`.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..aria.calibration import RectifiedCamera, Rectifier
from ..aria.mps import OnlineCalibrationProvider, Trajectory, find_mps_paths, read_trajectory_csv
from ..aria.vrs import AriaVrsReader
from ..map_format import Keyframe, MapMetadata, MapPaths, save_keyframes
from .keyframes import select_keyframes
from .reference_model import write_reference_model

logger = logging.getLogger(__name__)


@dataclass
class MapBuildConfig:
    """Parameters of :func:`build_map`."""

    vrs: Path
    mps: Path
    output: Path
    cameras: Optional[List[str]] = None  # None -> all RGB/SLAM streams found in the VRS
    # keyframe selection
    min_translation_m: float = 0.15
    min_rotation_deg: float = 10.0
    max_keyframes_per_camera: Optional[int] = None
    max_pose_gap_ms: float = 100.0
    min_quality: Optional[float] = None
    # rectification
    rectified_scale: float = 1.0  # image size relative to the source
    focal_scale: float = 1.0  # focal length relative to the source (smaller = wider FOV)
    upright: bool = True
    use_online_calibration: bool = True
    apply_time_offset: bool = True
    image_format: str = "jpg"
    jpeg_quality: int = 95
    # hloc
    feature_conf: str = "superpoint_aachen"
    matcher_conf: str = "superpoint+lightglue"
    retrieval_conf: str = "netvlad"
    pairs_from: str = "poses"  # poses | retrieval | both
    num_pairs: int = 20
    pairs_rotation_threshold_deg: float = 45.0
    skip_geometric_verification: bool = False
    verbose: bool = False
    overwrite: bool = False
    # skip the hloc stage (only extract keyframes and the reference model)
    keyframes_only: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k in ("vrs", "mps", "output"):
            d[k] = str(d[k])
        return d


@dataclass
class KeyframeExtractionResult:
    cameras: Dict[str, RectifiedCamera]
    keyframes: List[Keyframe]
    stats: Dict = field(default_factory=dict)


def _save_image(path: Path, image: np.ndarray, fmt: str, quality: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    params = []
    if fmt.lower() in ("jpg", "jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if not cv2.imwrite(str(path), image, params):
        raise IOError(f"Failed to write {path}")


def extract_keyframes(
    cfg: MapBuildConfig,
    reader: AriaVrsReader,
    trajectory: Trajectory,
    online_calibration: Optional[OnlineCalibrationProvider] = None,
) -> KeyframeExtractionResult:
    """Select, rectify and store keyframes for every requested camera stream."""
    paths = MapPaths(cfg.output)
    labels = reader.select_labels(cfg.cameras)
    max_gap_ns = int(cfg.max_pose_gap_ms * 1e6)
    cameras: Dict[str, RectifiedCamera] = {}
    keyframes: List[Keyframe] = []
    stats: Dict = {"per_camera": {}}

    for label in labels:
        factory_calib = reader.camera_calibration(label)
        src_w, src_h = [int(v) for v in factory_calib.get_image_size()]
        width = int(round(src_w * cfg.rectified_scale))
        height = int(round(src_h * cfg.rectified_scale))
        focal = float(np.asarray(factory_calib.get_focal_lengths()).reshape(-1)[0]) * cfg.rectified_scale * cfg.focal_scale
        base = Rectifier(factory_calib, width=width, height=height, focal=focal, upright=cfg.upright)
        cameras[label] = base.camera
        time_offset_ns = 0
        if cfg.apply_time_offset:
            try:
                time_offset_ns = int(round(float(factory_calib.get_time_offset_sec_device_camera()) * 1e9))
            except Exception:  # pragma: no cover
                time_offset_ns = 0

        timestamps = reader.timestamps_ns(label)
        poses = []
        for ts in timestamps:
            p = trajectory.pose_at(int(ts) + time_offset_ns, max_gap_ns=max_gap_ns, min_quality=cfg.min_quality)
            poses.append(None if p is None else p.T_world_device)
        num_posed = sum(p is not None for p in poses)
        selected = select_keyframes(poses, cfg.min_translation_m, cfg.min_rotation_deg, cfg.max_keyframes_per_camera)
        logger.info(
            "%s: %d frames, %d with a pose, %d keyframes selected", label, len(timestamps), num_posed, len(selected)
        )

        t0 = time.time()
        for n, idx in enumerate(selected):
            frame = reader.get_frame(label, idx)
            ts_pose = frame.timestamp_ns + time_offset_ns
            pose = trajectory.pose_at(ts_pose, max_gap_ns=max_gap_ns, min_quality=cfg.min_quality)
            if pose is None:
                continue
            rectifier = base
            if online_calibration is not None and cfg.use_online_calibration:
                online = online_calibration.camera_calibration(label, frame.timestamp_ns)
                if online is not None:
                    rectifier = Rectifier(online, width=width, height=height, focal=focal, upright=cfg.upright)
            rect = rectifier(frame.image)
            name = f"{label}/{frame.timestamp_ns}.{cfg.image_format}"
            _save_image(paths.images / name, rect, cfg.image_format, cfg.jpeg_quality)
            T_world_camera = pose.T_world_device @ rectifier.camera.T_device_camera
            keyframes.append(Keyframe(name, label, frame.timestamp_ns, T_world_camera, pose.T_world_device))
            if (n + 1) % 100 == 0:
                logger.info("  %s: wrote %d/%d keyframes (%.1f s)", label, n + 1, len(selected), time.time() - t0)
        stats["per_camera"][label] = {
            "num_frames": int(len(timestamps)),
            "num_posed": int(num_posed),
            "num_keyframes": int(len(selected)),
            "rectified_size": [cameras[label].pinhole.width, cameras[label].pinhole.height],
        }
    stats["num_keyframes"] = len(keyframes)
    return KeyframeExtractionResult(cameras, keyframes, stats)


def _merge_pairs(files: List[Path], output: Path) -> None:
    seen = set()
    lines = []
    for f in files:
        with open(f) as fd:
            for line in fd:
                parts = line.split()
                if len(parts) != 2:
                    continue
                key = tuple(parts)
                if key in seen or (key[1], key[0]) in seen:
                    continue
                seen.add(key)
                lines.append(" ".join(key))
    with open(output, "w") as fd:
        fd.write("\n".join(lines))


def run_hloc_pipeline(cfg: MapBuildConfig, paths: MapPaths) -> Dict:
    """Run the hloc feature extraction / matching / triangulation on the keyframes."""
    from hloc import extract_features, match_features, pairs_from_poses, pairs_from_retrieval, triangulation

    feature_conf = extract_features.confs[cfg.feature_conf]
    retrieval_conf = extract_features.confs[cfg.retrieval_conf]
    matcher_conf = match_features.confs[cfg.matcher_conf]

    logger.info("Extracting global descriptors (%s)", cfg.retrieval_conf)
    extract_features.main(retrieval_conf, paths.images, feature_path=paths.global_features, overwrite=cfg.overwrite)

    logger.info("Selecting image pairs from %s", cfg.pairs_from)
    pair_files = []
    if cfg.pairs_from in ("poses", "both"):
        p = paths.root / "pairs-poses.txt"
        pairs_from_poses.main(paths.reference, p, cfg.num_pairs, rotation_threshold=cfg.pairs_rotation_threshold_deg)
        pair_files.append(p)
    if cfg.pairs_from in ("retrieval", "both"):
        p = paths.root / "pairs-retrieval.txt"
        pairs_from_retrieval.main(paths.global_features, p, cfg.num_pairs)
        pair_files.append(p)
    if not pair_files:
        raise ValueError(f"Unknown pairs_from={cfg.pairs_from!r} (poses|retrieval|both)")
    _merge_pairs(pair_files, paths.pairs)

    logger.info("Extracting local features (%s)", cfg.feature_conf)
    extract_features.main(feature_conf, paths.images, feature_path=paths.features, overwrite=cfg.overwrite)

    logger.info("Matching (%s)", cfg.matcher_conf)
    match_features.main(matcher_conf, paths.pairs, paths.features, matches=paths.matches, overwrite=cfg.overwrite)

    logger.info("Triangulating with the MPS poses")
    if paths.sfm.exists() and cfg.overwrite:
        shutil.rmtree(paths.sfm)
    reconstruction = triangulation.main(
        paths.sfm,
        paths.reference,
        paths.images,
        paths.pairs,
        paths.features,
        paths.matches,
        skip_geometric_verification=cfg.skip_geometric_verification,
        verbose=cfg.verbose,
    )
    num_with_points = sum(1 for im in reconstruction.images.values() if im.num_points3D > 0)
    stats = {
        "num_images": int(len(reconstruction.images)),
        "num_images_with_points": int(num_with_points),
        "num_points3D": int(len(reconstruction.points3D)),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()),
        "mean_observations_per_image": float(reconstruction.compute_mean_observations_per_reg_image()),
    }
    logger.info("Triangulation statistics: %s", json.dumps(stats))
    return stats


def build_map(cfg: MapBuildConfig) -> MapPaths:
    """Build a complete map directory (see :mod:`aria_hloc.map_format`)."""
    t_start = time.time()
    paths = MapPaths(cfg.output)
    if paths.root.exists() and cfg.overwrite:
        logger.warning("Removing existing map directory %s", paths.root)
        shutil.rmtree(paths.root)
    paths.root.mkdir(parents=True, exist_ok=True)

    mps_paths = find_mps_paths(Path(cfg.mps))
    trajectory = read_trajectory_csv(mps_paths.closed_loop_trajectory)
    online = None
    if cfg.use_online_calibration and mps_paths.online_calibration is not None:
        try:
            online = OnlineCalibrationProvider(mps_paths.online_calibration)
        except Exception as e:  # pragma: no cover - depends on MPS version
            logger.warning("Could not read online calibration (%s); using factory calibration", e)

    reader = AriaVrsReader(Path(cfg.vrs))
    logger.info("Opened %s (device %s, cameras %s)", reader.path.name, reader.device_version, reader.camera_labels())

    extraction = extract_keyframes(cfg, reader, trajectory, online)
    if not extraction.keyframes:
        raise RuntimeError("No keyframes extracted: check that the VRS and MPS trajectory belong together")
    save_keyframes(extraction.keyframes, paths.root)
    write_reference_model(extraction.cameras, extraction.keyframes, paths.reference)

    meta = MapMetadata(
        cameras=extraction.cameras,
        feature_conf=cfg.feature_conf,
        matcher_conf=cfg.matcher_conf,
        retrieval_conf=cfg.retrieval_conf,
        source={
            "vrs": str(Path(cfg.vrs).name),
            "mps_trajectory": str(mps_paths.closed_loop_trajectory),
            "graph_uids": trajectory.graph_uids(),
            "device_version": reader.device_version,
            "build_config": cfg.to_dict(),
        },
        stats=extraction.stats,
    )
    meta.save(paths.root)

    if not cfg.keyframes_only:
        meta.stats["hloc"] = run_hloc_pipeline(cfg, paths)
        meta.save(paths.root)
    meta.stats["build_time_s"] = round(time.time() - t_start, 1)
    meta.save(paths.root)
    logger.info("Map written to %s in %.1f s", paths.root, time.time() - t_start)
    return paths
