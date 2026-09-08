"""Localize single images in an Aria-Hloc map.

The :class:`Relocalizer` keeps the triangulated COLMAP model, the keyframe
features and the retrieval descriptors in memory and answers queries with the
same algorithm as ``hloc.localize_sfm`` (retrieval -> matching against the
retrieved keyframes -> 2D-3D correspondences -> PnP + RANSAC + refinement),
without touching the file system per query.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..aria.calibration import PinholeCamera, RectifiedCamera, Rectifier
from ..geometry import invert_transform, transform_to_dict
from ..map_format import MapMetadata, MapPaths

logger = logging.getLogger(__name__)


@dataclass
class RelocalizerConfig:
    num_retrieved: int = 10
    ransac_max_error_px: float = 12.0
    min_num_inliers: int = 12
    min_inlier_ratio: float = 0.0
    covisibility_clustering: bool = False
    # restrict retrieval to keyframes of the same camera label as the query
    same_camera_only: bool = False
    device: Optional[str] = None  # auto
    feature_cache_size: int = 512


@dataclass
class LocalizationResult:
    success: bool
    T_world_camera: Optional[np.ndarray] = None
    T_world_device: Optional[np.ndarray] = None
    num_inliers: int = 0
    num_matches: int = 0
    num_keypoints: int = 0
    inlier_ratio: float = 0.0
    retrieved: List[str] = field(default_factory=list)
    camera_label: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["T_world_camera"] = None if self.T_world_camera is None else transform_to_dict(self.T_world_camera)
        d["T_world_device"] = None if self.T_world_device is None else transform_to_dict(self.T_world_device)
        return d


class _LRU:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._d: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)


def _rigid3d_to_matrix(rigid) -> np.ndarray:
    M = rigid.matrix() if callable(getattr(rigid, "matrix", None)) else np.asarray(rigid)
    M = np.asarray(M, dtype=np.float64)
    T = np.eye(4)
    T[:3, :4] = M[:3, :4]
    return T


class Relocalizer:
    """Localize images against a map built by :func:`aria_hloc.mapping.build_map`."""

    def __init__(self, map_dir: Path, config: Optional[RelocalizerConfig] = None, load_models: bool = True):
        import h5py
        import pycolmap

        self.paths = MapPaths(map_dir)
        self.config = config or RelocalizerConfig()
        self.meta = MapMetadata.load(self.paths.root)
        if not self.paths.is_complete():
            raise FileNotFoundError(f"Map at {self.paths.root} is incomplete (missing sfm/features)")

        t0 = time.time()
        self.reconstruction = pycolmap.Reconstruction(str(self.paths.sfm))
        self.db_name_to_id = {img.name: i for i, img in self.reconstruction.images.items()}
        # Global descriptors of the keyframes for retrieval.
        with h5py.File(str(self.paths.global_features), "r", libver="latest") as fd:
            names = [n for n in self.db_name_to_id if n in fd]
            desc = np.stack([fd[n]["global_descriptor"].__array__() for n in names], 0).astype(np.float32)
        self.db_names = names
        self.db_labels = np.array([n.split("/")[0] for n in names])
        self.db_descriptors = desc / np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-9)
        self._feature_cache = _LRU(self.config.feature_cache_size)
        self._h5_lock = threading.Lock()
        # Per keyframe: point3D id of every keypoint (-1 when not triangulated).
        self._points3D_ids: Dict[int, np.ndarray] = {}
        self._rectifiers: Dict[str, Rectifier] = {}
        self._lock = threading.Lock()
        logger.info(
            "Loaded map %s: %d keyframes, %d 3D points, cameras %s (%.1f s)",
            self.paths.root,
            len(self.db_names),
            len(self.reconstruction.points3D),
            list(self.meta.cameras),
            time.time() - t0,
        )
        self.extractor = self.retrieval = self.matcher = None
        if load_models:
            self.load_models()

    # ------------------------------------------------------------------ setup
    def load_models(self) -> None:
        from .hloc_models import FeatureMatcher, GlobalDescriptorExtractor, LocalFeatureExtractor

        t0 = time.time()
        self.extractor = LocalFeatureExtractor(self.meta.feature_conf, self.config.device)
        self.retrieval = GlobalDescriptorExtractor(self.meta.retrieval_conf, self.config.device)
        self.matcher = FeatureMatcher(self.meta.matcher_conf, self.config.device)
        logger.info("Loaded hloc models on %s (%.1f s)", self.extractor.device, time.time() - t0)

    @property
    def cameras(self) -> Dict[str, RectifiedCamera]:
        return self.meta.cameras

    def rectifier(self, label: str, src_calib=None) -> Rectifier:
        """Rectifier reproducing the map's pinhole camera for ``label``.

        When ``src_calib`` (a ``projectaria_tools`` ``CameraCalibration`` of the
        query device) is given it replaces the stored calibration, e.g. when the
        query frames come from a different Aria device.
        """
        if label not in self.meta.cameras:
            raise KeyError(f"Camera {label!r} not in map; available: {list(self.meta.cameras)}")
        if src_calib is not None:
            return Rectifier.from_rectified_camera(self.meta.cameras[label], src_calib)
        with self._lock:
            if label not in self._rectifiers:
                self._rectifiers[label] = Rectifier.from_rectified_camera(self.meta.cameras[label])
            return self._rectifiers[label]

    # --------------------------------------------------------------- features
    def _db_features(self, name: str) -> Dict[str, np.ndarray]:
        feats = self._feature_cache.get(name)
        if feats is not None:
            return feats
        import h5py

        with self._h5_lock:
            with h5py.File(str(self.paths.features), "r", libver="latest") as fd:
                grp = fd[name]
                feats = {k: grp[k].__array__() for k in grp.keys()}
        feats = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in feats.items()}
        self._feature_cache.put(name, feats)
        return feats

    def _db_points3D_ids(self, image_id: int) -> np.ndarray:
        ids = self._points3D_ids.get(image_id)
        if ids is None:
            image = self.reconstruction.images[image_id]
            ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D], dtype=np.int64)
            with self._lock:
                self._points3D_ids[image_id] = ids
        return ids

    def retrieve(self, image: np.ndarray, num: Optional[int] = None, label: Optional[str] = None) -> List[str]:
        """Return the names of the ``num`` most similar keyframes."""
        q = self.retrieval(image)
        q = q / max(np.linalg.norm(q), 1e-9)
        scores = self.db_descriptors @ q
        if label is not None and self.config.same_camera_only:
            scores = np.where(self.db_labels == label, scores, -np.inf)
        num = num or self.config.num_retrieved
        order = np.argsort(-scores)[:num]
        return [self.db_names[i] for i in order if np.isfinite(scores[i])]

    # ------------------------------------------------------------- localize
    def localize_image(
        self,
        image: np.ndarray,
        camera: PinholeCamera,
        camera_label: Optional[str] = None,
        num_retrieved: Optional[int] = None,
    ) -> LocalizationResult:
        """Localize an already rectified pinhole image.

        Args:
            image: HxWx3 RGB or HxW grayscale uint8 image.
            camera: pinhole intrinsics of ``image``.
            camera_label: Aria camera label; when it is one of the map cameras
                the device pose ``T_world_device`` is derived from the stored
                ``T_device_camera``.
        Returns:
            :class:`LocalizationResult` with ``T_world_camera`` (camera-to-world).
        """
        if self.extractor is None:
            self.load_models()
        timings: Dict[str, float] = {}
        t = time.time()
        retrieved = self.retrieve(image, num_retrieved, camera_label)
        timings["retrieval"] = (time.time() - t) * 1e3
        t = time.time()
        qfeats = self.extractor(image)
        timings["features"] = (time.time() - t) * 1e3
        result = self._localize_from_features(qfeats, camera, retrieved, timings)
        result.camera_label = camera_label
        if result.success and camera_label in self.meta.cameras:
            T_device_camera = self.meta.cameras[camera_label].T_device_camera
            result.T_world_device = result.T_world_camera @ invert_transform(T_device_camera)
        return result

    def localize_aria_frame(
        self,
        image: np.ndarray,
        camera_label: str,
        src_calib=None,
        num_retrieved: Optional[int] = None,
    ) -> LocalizationResult:
        """Localize a raw (fisheye) Aria frame of camera ``camera_label``.

        The frame is rectified exactly like the map keyframes. ``src_calib``
        optionally provides the query device's own ``CameraCalibration`` (from
        its VRS or serialized with :func:`aria_hloc.aria.calibration.camera_calibration_to_dict`).
        """
        t = time.time()
        rectifier = self.rectifier(camera_label, src_calib)
        rect = rectifier(image)
        cam = rectifier.camera
        timings = {"rectify": (time.time() - t) * 1e3}
        result = self.localize_image(rect, cam.pinhole, camera_label, num_retrieved)
        if result.success:
            # Use the (possibly device specific) extrinsics of the rectifier.
            result.T_world_device = result.T_world_camera @ invert_transform(cam.T_device_camera)
        result.timings_ms = {**timings, **result.timings_ms}
        return result

    def _localize_from_features(
        self,
        qfeats: Dict[str, np.ndarray],
        camera: PinholeCamera,
        retrieved: Sequence[str],
        timings: Dict[str, float],
    ) -> LocalizationResult:
        import pycolmap

        db_ids = [self.db_name_to_id[n] for n in retrieved if n in self.db_name_to_id]
        result = LocalizationResult(False, retrieved=list(retrieved), num_keypoints=int(len(qfeats["keypoints"])))
        if not db_ids:
            result.message = "no keyframes retrieved"
            result.timings_ms = timings
            return result

        clusters: List[List[int]]
        if self.config.covisibility_clustering:
            from hloc.localize_sfm import do_covisibility_clustering

            clusters = do_covisibility_clustering(db_ids, self.reconstruction)
        else:
            clusters = [db_ids]

        t = time.time()
        # Match once against every retrieved keyframe, reuse for all clusters.
        matches_by_db: Dict[int, np.ndarray] = {}
        for db_id in db_ids:
            name = self.reconstruction.images[db_id].name
            m, _ = self.matcher(qfeats, self._db_features(name))
            matches_by_db[db_id] = m
        timings["matching"] = (time.time() - t) * 1e3

        t = time.time()
        kpq = qfeats["keypoints"].astype(np.float64) + 0.5  # COLMAP pixel convention
        best = None
        for cluster in clusters:
            kp_idx_to_3D: Dict[int, List[int]] = defaultdict(list)
            num_matches = 0
            for db_id in cluster:
                p3d_ids = self._db_points3D_ids(db_id)
                m = matches_by_db[db_id]
                if len(m) == 0 or len(p3d_ids) == 0:
                    continue
                valid = m[:, 1] < len(p3d_ids)
                m = m[valid]
                m = m[p3d_ids[m[:, 1]] != -1]
                num_matches += len(m)
                for qi, di in m:
                    id_3D = int(p3d_ids[di])
                    if id_3D not in kp_idx_to_3D[qi]:
                        kp_idx_to_3D[qi].append(id_3D)
            idxs = list(kp_idx_to_3D.keys())
            mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
            mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
            if len(mkp_idxs) < 4:
                continue
            points2D = kpq[mkp_idxs]
            points3D = np.array([self.reconstruction.points3D[j].xyz for j in mp3d_ids], dtype=np.float64)
            ret = pycolmap.estimate_and_refine_absolute_pose(
                points2D,
                points3D,
                camera.to_pycolmap(),
                estimation_options={"ransac": {"max_error": self.config.ransac_max_error_px}},
                refinement_options={},
            )
            if ret is None:
                continue
            num_inliers = int(ret["num_inliers"])
            if best is None or num_inliers > best[0]:
                best = (num_inliers, ret, num_matches, len(mkp_idxs))
        timings["pnp"] = (time.time() - t) * 1e3
        result.timings_ms = timings

        if best is None:
            result.message = "PnP failed"
            return result
        num_inliers, ret, num_matches, num_corr = best
        result.num_inliers = num_inliers
        result.num_matches = int(num_matches)
        result.inlier_ratio = float(num_inliers / max(num_corr, 1))
        T_cam_world = _rigid3d_to_matrix(ret["cam_from_world"])
        result.T_world_camera = invert_transform(T_cam_world)
        if num_inliers < self.config.min_num_inliers or result.inlier_ratio < self.config.min_inlier_ratio:
            result.message = f"too few inliers ({num_inliers}, ratio {result.inlier_ratio:.2f})"
            return result
        result.success = True
        return result

    def info(self) -> Dict[str, Any]:
        return {
            "map_dir": str(self.paths.root),
            "num_keyframes": len(self.db_names),
            "num_points3D": len(self.reconstruction.points3D),
            "cameras": {k: v.to_dict() for k, v in self.meta.cameras.items()},
            "feature_conf": self.meta.feature_conf,
            "matcher_conf": self.meta.matcher_conf,
            "retrieval_conf": self.meta.retrieval_conf,
            "source": self.meta.source,
            "stats": self.meta.stats,
            "config": asdict(self.config),
        }
