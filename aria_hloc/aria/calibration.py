"""Camera calibration handling for Aria cameras.

Aria cameras (RGB and SLAM, Gen 1 and Gen 2) use the ``Fisheye624`` model,
which COLMAP/hloc do not support. We therefore rectify every image to a
pinhole (``LINEAR``) camera with ``projectaria_tools`` and run the whole
mapping/localization pipeline on pinhole images. The pinhole camera keeps the
optical centre and the optical axis of the fisheye camera, so the extrinsics
``T_device_camera`` stay valid (optionally composed with a 90 degree rotation
when images are made upright).

Everything that needs ``projectaria_tools`` is imported lazily so the pure
numpy data structures can be used (and unit tested) without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..geometry import transform_from_dict, transform_to_dict


@dataclass
class PinholeCamera:
    """A pinhole camera (COLMAP ``PINHOLE`` model)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def colmap_params(self):
        return [float(self.fx), float(self.fy), float(self.cx), float(self.cy)]

    def to_pycolmap(self, camera_id: Optional[int] = None):
        import pycolmap

        kwargs = dict(model="PINHOLE", width=int(self.width), height=int(self.height), params=self.colmap_params)
        if camera_id is not None:
            kwargs["camera_id"] = camera_id
        return pycolmap.Camera(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "PINHOLE",
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PinholeCamera":
        if "params" in d:  # COLMAP style
            fx, fy, cx, cy = d["params"][:4]
            return cls(int(d["width"]), int(d["height"]), fx, fy, cx, cy)
        return cls(int(d["width"]), int(d["height"]), d["fx"], d["fy"], d["cx"], d["cy"])

    def scaled(self, factor: float) -> "PinholeCamera":
        """Camera of the image resized by ``factor`` (COLMAP pixel-centre convention)."""
        w = int(round(self.width * factor))
        h = int(round(self.height * factor))
        sx, sy = w / self.width, h / self.height
        return PinholeCamera(w, h, self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)


@dataclass
class RectifiedCamera:
    """Description of a rectified (pinhole) Aria camera stored with a map.

    Attributes:
        label: Aria camera label, e.g. ``camera-rgb`` or ``camera-slam-front-left``.
        pinhole: intrinsics of the rectified image.
        T_device_camera: extrinsics of the rectified camera in the device frame.
        upright: whether the rectified image was rotated 90 degrees clockwise.
        source: serialized fisheye ``CameraCalibration`` used to rectify (may be
            empty when the map was built from already-rectified images).
    """

    label: str
    pinhole: PinholeCamera
    T_device_camera: np.ndarray
    upright: bool = True
    source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "pinhole": self.pinhole.to_dict(),
            "T_device_camera": transform_to_dict(self.T_device_camera),
            "upright": bool(self.upright),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RectifiedCamera":
        return cls(
            d["label"],
            PinholeCamera.from_dict(d["pinhole"]),
            transform_from_dict(d["T_device_camera"]),
            bool(d.get("upright", True)),
            d.get("source", {}) or {},
        )


# ---------------------------------------------------------------------------
# projectaria_tools dependent helpers
# ---------------------------------------------------------------------------


def se3_to_matrix(T) -> np.ndarray:
    """Convert a ``projectaria_tools`` ``SE3`` (possibly batched of size 1) to 4x4."""
    M = np.asarray(T.to_matrix(), dtype=np.float64)
    return M.reshape(-1, 4, 4)[0]


def matrix_to_se3(M):
    from projectaria_tools.core.sophus import SE3

    return SE3.from_matrix(np.asarray(M, dtype=np.float64).reshape(4, 4))


def camera_calibration_to_dict(calib) -> Dict[str, Any]:
    """Serialize a ``projectaria_tools`` ``CameraCalibration`` to plain JSON types."""
    size = calib.get_image_size()
    valid_radius = calib.get_valid_radius()
    d = {
        "label": calib.get_label(),
        "model": calib.get_model_name().name,
        "projection_params": [float(v) for v in np.asarray(calib.get_projection_params()).reshape(-1)],
        "T_device_camera": transform_to_dict(se3_to_matrix(calib.get_transform_device_camera())),
        "width": int(size[0]),
        "height": int(size[1]),
        "valid_radius": None if valid_radius is None else float(valid_radius),
        "max_solid_angle": float(calib.get_max_solid_angle()),
        "serial_number": calib.get_serial_number(),
    }
    try:
        d["time_offset_sec_device_camera"] = float(calib.get_time_offset_sec_device_camera())
    except Exception:  # pragma: no cover - older projectaria_tools
        pass
    return d


def camera_calibration_from_dict(d: Dict[str, Any]):
    """Rebuild a ``projectaria_tools`` ``CameraCalibration`` from :func:`camera_calibration_to_dict`."""
    from projectaria_tools.core import calibration as aria_calib

    model = getattr(aria_calib.CameraModelType, d["model"])
    params = np.asarray(d["projection_params"], dtype=np.float64)
    T = matrix_to_se3(transform_from_dict(d["T_device_camera"]))
    args = [
        d["label"],
        model,
        params,
        T,
        int(d["width"]),
        int(d["height"]),
        d.get("valid_radius"),
        float(d.get("max_solid_angle", np.pi)),
        d.get("serial_number", ""),
    ]
    if "time_offset_sec_device_camera" in d:
        try:
            return aria_calib.CameraCalibration(*args, float(d["time_offset_sec_device_camera"]))
        except TypeError:  # pragma: no cover
            pass
    return aria_calib.CameraCalibration(*args)


class Rectifier:
    """Rectify raw (fisheye) Aria images to a pinhole camera.

    Args:
        src_calib: ``projectaria_tools`` ``CameraCalibration`` of the raw images.
        width/height: size of the rectified image (default: source size).
        focal: focal length of the rectified image in pixels (default: source focal).
        upright: rotate the rectified image 90 degrees clockwise so that it is
            upright (Aria sensors are mounted rotated). Learned feature
            extractors work best on upright images.
    """

    def __init__(
        self,
        src_calib,
        width: Optional[int] = None,
        height: Optional[int] = None,
        focal: Optional[float] = None,
        upright: bool = True,
    ):
        from projectaria_tools.core import calibration as aria_calib

        self.src_calib = src_calib
        src_w, src_h = [int(v) for v in src_calib.get_image_size()]
        self.width = int(width or src_w)
        self.height = int(height or src_h)
        if focal is None:
            focal = float(np.asarray(src_calib.get_focal_lengths()).reshape(-1)[0])
            focal *= self.width / src_w
        self.focal = float(focal)
        self.upright = bool(upright)
        self.label = src_calib.get_label()
        self._linear = aria_calib.get_linear_camera_calibration(
            self.width, self.height, self.focal, self.label, src_calib.get_transform_device_camera()
        )
        self._final = aria_calib.rotate_camera_calib_cw90deg(self._linear) if self.upright else self._linear

    @property
    def camera(self) -> RectifiedCamera:
        f = np.asarray(self._final.get_focal_lengths()).reshape(-1)
        c = np.asarray(self._final.get_principal_point()).reshape(-1)
        size = self._final.get_image_size()
        pinhole = PinholeCamera(int(size[0]), int(size[1]), float(f[0]), float(f[1]), float(c[0]), float(c[1]))
        return RectifiedCamera(
            label=self.label,
            pinhole=pinhole,
            T_device_camera=se3_to_matrix(self._final.get_transform_device_camera()),
            upright=self.upright,
            source=camera_calibration_to_dict(self.src_calib),
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from projectaria_tools.core import calibration as aria_calib

        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        src_w, src_h = [int(v) for v in self.src_calib.get_image_size()]
        if image.shape[1] != src_w or image.shape[0] != src_h:
            raise ValueError(
                f"{self.label}: image is {image.shape[1]}x{image.shape[0]} but the calibration expects "
                f"{src_w}x{src_h}; rescale the calibration (CameraCalibration.rescale) or pass matching frames"
            )
        rect = aria_calib.distort_by_calibration(image, self._linear, self.src_calib)
        rect = np.asarray(rect)
        if self.upright:
            rect = np.ascontiguousarray(np.rot90(rect, k=-1))
        return rect

    @classmethod
    def from_rectified_camera(cls, cam: RectifiedCamera, src_calib=None) -> "Rectifier":
        """Recreate the rectifier used for a map camera (optionally with a new source calibration)."""
        if src_calib is None:
            if not cam.source:
                raise ValueError(f"Camera {cam.label} has no stored source calibration")
            src_calib = camera_calibration_from_dict(cam.source)
        # The stored pinhole is for the (possibly rotated) final image; undo the rotation for sizes.
        w, h = (cam.pinhole.height, cam.pinhole.width) if cam.upright else (cam.pinhole.width, cam.pinhole.height)
        return cls(src_calib, width=w, height=h, focal=cam.pinhole.fx, upright=cam.upright)
