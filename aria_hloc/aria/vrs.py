"""Minimal VRS frame reader built on ``projectaria_tools``.

Works for Aria Gen 1 and Gen 2 recordings. Camera labels are discovered from
the device calibration: Gen 1 ``camera-rgb``, ``camera-slam-left``,
``camera-slam-right``; Gen 2 ``camera-rgb``, ``slam-front-left``,
``slam-front-right``, ``slam-side-left``, ``slam-side-right``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MAPPING_CAMERAS = [
    "camera-rgb",
    # Gen 2 labels (as reported by the device calibration of Gen 2 recordings)
    "slam-front-left",
    "slam-front-right",
    "slam-side-left",
    "slam-side-right",
    # Gen 1 labels
    "camera-slam-left",
    "camera-slam-right",
]


@dataclass
class AriaFrame:
    """A raw image from a VRS stream (device time nanoseconds)."""

    label: str
    index: int
    timestamp_ns: int
    image: np.ndarray  # HxW (grayscale) or HxWx3 (RGB) uint8


class AriaVrsReader:
    """Iterate over the image streams of a VRS file."""

    def __init__(self, path: Path):
        from projectaria_tools.core import data_provider

        self.path = Path(path)
        self.provider = data_provider.create_vrs_data_provider(str(self.path))
        if self.provider is None:
            raise IOError(f"Could not open VRS file {self.path}")
        self.device_calibration = self.provider.get_device_calibration()
        if self.device_calibration is None:
            raise IOError(f"VRS file {self.path} has no device calibration")
        self._stream_ids = {}
        for label in self.device_calibration.get_camera_labels():
            sid = self.provider.get_stream_id_from_label(label)
            if sid is None:
                continue
            try:
                n = self.provider.get_num_data(sid)
            except Exception:  # pragma: no cover
                n = 0
            if n > 0:
                self._stream_ids[label] = sid

    @property
    def device_version(self) -> str:
        try:
            return str(self.device_calibration.get_device_version()).split(".")[-1]
        except Exception:  # pragma: no cover
            return "unknown"

    def camera_labels(self, include_eye_tracking: bool = False) -> List[str]:
        labels = [l for l in self._stream_ids if include_eye_tracking or not l.startswith("camera-et")]
        return sorted(labels)

    def select_labels(self, requested: Optional[List[str]] = None) -> List[str]:
        """Resolve requested labels against the streams present in the file."""
        available = self.camera_labels()
        if not requested:
            chosen = [l for l in DEFAULT_MAPPING_CAMERAS if l in available]
        else:
            missing = [l for l in requested if l not in available]
            if missing:
                raise ValueError(f"Camera streams {missing} not found in {self.path.name}; available: {available}")
            chosen = list(requested)
        if not chosen:
            raise ValueError(f"No usable camera streams in {self.path.name}; available: {available}")
        return chosen

    def camera_calibration(self, label: str):
        calib = self.device_calibration.get_camera_calib(label)
        if calib is None:
            raise ValueError(f"No calibration for camera {label}")
        return calib

    def num_frames(self, label: str) -> int:
        return int(self.provider.get_num_data(self._stream_ids[label]))

    def timestamps_ns(self, label: str) -> np.ndarray:
        from projectaria_tools.core.sensor_data import TimeDomain

        ts = self.provider.get_timestamps_ns(self._stream_ids[label], TimeDomain.DEVICE_TIME)
        return np.asarray(ts, dtype=np.int64)

    def get_frame(self, label: str, index: int) -> AriaFrame:
        sid = self._stream_ids[label]
        image_data, record = self.provider.get_image_data_by_index(sid, int(index))
        if not image_data.is_valid():
            raise ValueError(f"Invalid image {label}[{index}]")
        img = np.asarray(image_data.to_numpy_array())
        return AriaFrame(label, int(index), int(record.capture_timestamp_ns), img)

    def get_frame_at(self, label: str, timestamp_ns: int) -> AriaFrame:
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

        sid = self._stream_ids[label]
        image_data, record = self.provider.get_image_data_by_time_ns(
            sid, int(timestamp_ns), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )
        if not image_data.is_valid():
            raise ValueError(f"No image in {label} near {timestamp_ns}")
        img = np.asarray(image_data.to_numpy_array())
        return AriaFrame(label, -1, int(record.capture_timestamp_ns), img)

    def iter_frames(self, label: str, step: int = 1, start: int = 0, stop: Optional[int] = None) -> Iterator[AriaFrame]:
        n = self.num_frames(label)
        stop = n if stop is None else min(stop, n)
        for i in range(start, stop, max(1, step)):
            yield self.get_frame(label, i)
