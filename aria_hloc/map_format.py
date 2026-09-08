"""On-disk layout and metadata of an Aria-Hloc map.

::

    <map_dir>/
      map.json                 metadata (this module)
      images/<label>/<ts>.jpg  rectified keyframe images
      reference/               COLMAP model with MPS poses (no 3D points)
      sfm/                     triangulated COLMAP model used for localization
      features.h5              local features of the keyframes (hloc format)
      global-features.h5       global descriptors of the keyframes (hloc format)
      pairs.txt, matches.h5    intermediate hloc files
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .aria.calibration import RectifiedCamera
from .geometry import transform_from_dict, transform_to_dict

MAP_FORMAT_VERSION = 1

IMAGES_DIR = "images"
REFERENCE_DIR = "reference"
SFM_DIR = "sfm"
FEATURES_FILE = "features.h5"
GLOBAL_FEATURES_FILE = "global-features.h5"
PAIRS_FILE = "pairs.txt"
MATCHES_FILE = "matches.h5"
MAP_FILE = "map.json"
KEYFRAMES_FILE = "keyframes.json"


@dataclass
class Keyframe:
    name: str  # image name relative to images/, e.g. camera-rgb/123456789.jpg
    label: str
    timestamp_ns: int
    T_world_camera: np.ndarray
    T_world_device: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "label": self.label,
            "timestamp_ns": int(self.timestamp_ns),
            "T_world_camera": transform_to_dict(self.T_world_camera),
        }
        if self.T_world_device is not None:
            d["T_world_device"] = transform_to_dict(self.T_world_device)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Keyframe":
        return cls(
            d["name"],
            d["label"],
            int(d["timestamp_ns"]),
            transform_from_dict(d["T_world_camera"]),
            transform_from_dict(d["T_world_device"]) if "T_world_device" in d else None,
        )


@dataclass
class MapMetadata:
    cameras: Dict[str, RectifiedCamera]
    feature_conf: str = "superpoint_aachen"
    matcher_conf: str = "superpoint+lightglue"
    retrieval_conf: str = "netvlad"
    source: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    format_version: int = MAP_FORMAT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "feature_conf": self.feature_conf,
            "matcher_conf": self.matcher_conf,
            "retrieval_conf": self.retrieval_conf,
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "source": self.source,
            "stats": self.stats,
            "layout": {
                "images": IMAGES_DIR,
                "reference": REFERENCE_DIR,
                "sfm": SFM_DIR,
                "features": FEATURES_FILE,
                "global_features": GLOBAL_FEATURES_FILE,
                "pairs": PAIRS_FILE,
                "matches": MATCHES_FILE,
                "keyframes": KEYFRAMES_FILE,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MapMetadata":
        return cls(
            cameras={k: RectifiedCamera.from_dict(v) for k, v in d["cameras"].items()},
            feature_conf=d.get("feature_conf", "superpoint_aachen"),
            matcher_conf=d.get("matcher_conf", "superpoint+lightglue"),
            retrieval_conf=d.get("retrieval_conf", "netvlad"),
            source=d.get("source", {}),
            stats=d.get("stats", {}),
            format_version=int(d.get("format_version", MAP_FORMAT_VERSION)),
        )

    def save(self, map_dir: Path) -> Path:
        path = Path(map_dir) / MAP_FILE
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, map_dir: Path) -> "MapMetadata":
        path = Path(map_dir) / MAP_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: is {map_dir} an Aria-Hloc map directory?")
        with open(path) as f:
            d = json.load(f)
        if int(d.get("format_version", 1)) > MAP_FORMAT_VERSION:
            raise ValueError(f"Map format {d['format_version']} is newer than supported {MAP_FORMAT_VERSION}")
        return cls.from_dict(d)


def save_keyframes(keyframes: List[Keyframe], map_dir: Path) -> Path:
    path = Path(map_dir) / KEYFRAMES_FILE
    with open(path, "w") as f:
        json.dump([k.to_dict() for k in keyframes], f)
    return path


def load_keyframes(map_dir: Path) -> List[Keyframe]:
    path = Path(map_dir) / KEYFRAMES_FILE
    if not path.exists():
        return []
    with open(path) as f:
        return [Keyframe.from_dict(d) for d in json.load(f)]


class MapPaths:
    """Absolute paths of the files of a map directory."""

    def __init__(self, map_dir: Path):
        self.root = Path(map_dir)
        self.map_file = self.root / MAP_FILE
        self.images = self.root / IMAGES_DIR
        self.reference = self.root / REFERENCE_DIR
        self.sfm = self.root / SFM_DIR
        self.features = self.root / FEATURES_FILE
        self.global_features = self.root / GLOBAL_FEATURES_FILE
        self.pairs = self.root / PAIRS_FILE
        self.matches = self.root / MATCHES_FILE
        self.keyframes = self.root / KEYFRAMES_FILE

    def is_complete(self) -> bool:
        return all(
            p.exists()
            for p in (self.map_file, self.features, self.global_features, self.sfm / "points3D.bin")
        )
