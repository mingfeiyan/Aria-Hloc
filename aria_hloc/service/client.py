"""Small HTTP client for the relocalization service."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def encode_image(image: Union[np.ndarray, Path, str, bytes], fmt: str = ".png") -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    import cv2

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(fmt, img)
    if not ok:
        raise ValueError("Could not encode image")
    return buf.tobytes()


class AriaHlocClient:
    """Client for :mod:`aria_hloc.service.app`.

    Example::

        client = AriaHlocClient("http://localhost:8080")
        res = client.localize(frame.image, camera_label="camera-rgb")
        if res["success"]:
            T_world_device = np.array(res["T_world_device"]["matrix"])
    """

    def __init__(self, url: str = "http://localhost:8080", timeout: float = 60.0):
        import httpx

        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=timeout)

    def health(self) -> Dict[str, Any]:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def map_info(self) -> Dict[str, Any]:
        r = self._client.get("/map")
        r.raise_for_status()
        return r.json()

    def localize(
        self,
        image: Union[np.ndarray, Path, str, bytes],
        camera_label: Optional[str] = None,
        rectified: bool = False,
        calibration: Optional[Dict[str, Any]] = None,
        num_retrieved: Optional[int] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {"rectified": "true" if rectified else "false"}
        if camera_label:
            data["camera_label"] = camera_label
        if calibration is not None:
            data["calibration"] = json.dumps(calibration)
        if num_retrieved is not None:
            data["num_retrieved"] = str(int(num_retrieved))
        files = {"image": ("query.png", io.BytesIO(encode_image(image)), "image/png")}
        r = self._client.post("/localize", data=data, files=files)
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()
