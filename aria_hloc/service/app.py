"""FastAPI application exposing a :class:`Relocalizer`.

Endpoints
---------
``GET /health``                 liveness + whether the map/models are loaded.
``GET /map``                    map metadata (cameras, stats, configuration).
``POST /localize``              multipart form: ``image`` file plus optional fields
                                ``camera_label``, ``rectified``, ``calibration``,
                                ``num_retrieved``.
``POST /localize/json``         same as above with a base64 encoded image in JSON.

Query images are either raw Aria frames (``rectified=false``, default), which
the service rectifies exactly like the map keyframes, or pinhole images
(``rectified=true``) whose intrinsics are given in ``calibration`` as
``{"width","height","fx","fy","cx","cy"}``. For raw frames ``calibration`` may
carry the query device's own Aria camera calibration (as produced by
:func:`aria_hloc.aria.calibration.camera_calibration_to_dict`); otherwise the
calibration stored with the map is used.
"""

import base64
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..aria.calibration import PinholeCamera, camera_calibration_from_dict
from ..localization.relocalizer import LocalizationResult, RelocalizerConfig

logger = logging.getLogger(__name__)


def decode_image(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to RGB (HxWx3) or grayscale (HxW) uint8."""
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = (img / 256).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
    return np.ascontiguousarray(img)


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


class LocalizationService:
    """Thread-safe wrapper used by the HTTP layer (and directly in tests)."""

    def __init__(self, relocalizer):
        self.relocalizer = relocalizer
        self._lock = threading.Lock()

    def localize(
        self,
        image: np.ndarray,
        camera_label: Optional[str] = None,
        rectified: bool = False,
        calibration: Optional[Dict[str, Any]] = None,
        num_retrieved: Optional[int] = None,
    ) -> LocalizationResult:
        if rectified:
            if calibration is None:
                if camera_label is None or camera_label not in self.relocalizer.cameras:
                    raise ValueError("rectified images need 'calibration' (pinhole intrinsics) or a map camera_label")
                pinhole = self.relocalizer.cameras[camera_label].pinhole
            else:
                pinhole = PinholeCamera.from_dict(calibration)
            with self._lock:
                return self.relocalizer.localize_image(image, pinhole, camera_label, num_retrieved)
        if camera_label is None:
            raise ValueError("raw Aria frames need 'camera_label' (e.g. camera-rgb)")
        src_calib = camera_calibration_from_dict(calibration) if calibration else None
        with self._lock:
            return self.relocalizer.localize_aria_frame(image, camera_label, src_calib, num_retrieved)


def create_app(
    map_dir: Optional[Path] = None,
    config: Optional[RelocalizerConfig] = None,
    relocalizer=None,
    lazy: bool = False,
):
    """Create the FastAPI app. Pass ``relocalizer`` to inject a preloaded (or fake) instance."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Aria-Hloc relocalization service", version="0.1.0")
    state: Dict[str, Any] = {"service": None, "map_dir": None if map_dir is None else str(map_dir), "error": None}
    if relocalizer is not None:
        state["service"] = LocalizationService(relocalizer)
    load_lock = threading.Lock()

    def get_service() -> LocalizationService:
        if state["service"] is None:
            with load_lock:
                if state["service"] is None:
                    if map_dir is None:
                        raise HTTPException(503, "No map configured")
                    from ..localization.relocalizer import Relocalizer

                    try:
                        state["service"] = LocalizationService(Relocalizer(Path(map_dir), config))
                    except Exception as e:  # pragma: no cover
                        state["error"] = str(e)
                        logger.exception("Failed to load map")
                        raise HTTPException(503, f"Failed to load map: {e}")
        return state["service"]

    if not lazy and relocalizer is None and map_dir is not None:
        get_service()

    @app.get("/health")
    def health():
        return {"status": "ok", "map_loaded": state["service"] is not None, "map_dir": state["map_dir"], "error": state["error"]}

    @app.get("/map")
    def map_info():
        return get_service().relocalizer.info()

    def _run(image_bytes: bytes, camera_label, rectified, calibration, num_retrieved):
        service = get_service()
        try:
            image = decode_image(image_bytes)
            calib = json.loads(calibration) if isinstance(calibration, str) and calibration.strip() else calibration
            result = service.localize(image, camera_label or None, _parse_bool(rectified), calib, num_retrieved)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        return JSONResponse(result.to_dict())

    @app.post("/localize")
    async def localize(
        image: UploadFile = File(...),
        camera_label: Optional[str] = Form(None),
        rectified: Optional[str] = Form("false"),
        calibration: Optional[str] = Form(None),
        num_retrieved: Optional[int] = Form(None),
    ):
        data = await image.read()
        return _run(data, camera_label, rectified, calibration, num_retrieved)

    @app.post("/localize/json")
    def localize_json(body: Dict[str, Any]):
        if "image_base64" not in body:
            raise HTTPException(400, "missing image_base64")
        try:
            data = base64.b64decode(body["image_base64"])
        except Exception:
            raise HTTPException(400, "invalid base64 image")
        return _run(
            data,
            body.get("camera_label"),
            body.get("rectified", False),
            body.get("calibration"),
            body.get("num_retrieved"),
        )

    return app


def serve(map_dir: Path, host: str = "0.0.0.0", port: int = 8080, config: Optional[RelocalizerConfig] = None, workers: int = 1):
    import uvicorn

    app = create_app(map_dir, config)
    uvicorn.run(app, host=host, port=port, workers=workers)
