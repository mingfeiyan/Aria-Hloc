import base64
import json

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")
from fastapi.testclient import TestClient  # noqa: E402

from aria_hloc.aria.calibration import PinholeCamera, RectifiedCamera  # noqa: E402
from aria_hloc.localization.relocalizer import LocalizationResult  # noqa: E402
from aria_hloc.service.app import create_app, decode_image  # noqa: E402
from aria_hloc.service.client import encode_image  # noqa: E402


class FakeRelocalizer:
    def __init__(self):
        self.cameras = {"camera-rgb": RectifiedCamera("camera-rgb", PinholeCamera(64, 48, 50, 50, 31.5, 23.5), np.eye(4))}
        self.calls = []

    def info(self):
        return {"num_keyframes": 3, "cameras": {k: v.to_dict() for k, v in self.cameras.items()}}

    def localize_image(self, image, camera, camera_label=None, num_retrieved=None):
        self.calls.append(("image", image.shape, camera, camera_label, num_retrieved))
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        return LocalizationResult(True, T, T, num_inliers=50, num_matches=80, retrieved=["camera-rgb/1.jpg"], camera_label=camera_label)

    def localize_aria_frame(self, image, camera_label, src_calib=None, num_retrieved=None):
        self.calls.append(("aria", image.shape, camera_label, src_calib, num_retrieved))
        return LocalizationResult(False, message="too few inliers", camera_label=camera_label)


@pytest.fixture
def client():
    fake = FakeRelocalizer()
    app = create_app(relocalizer=fake)
    return TestClient(app), fake


def png_bytes(shape=(48, 64, 3)):
    img = (np.random.default_rng(0).random(shape) * 255).astype(np.uint8)
    return encode_image(img), img


def test_health_and_map(client):
    c, _ = client
    assert c.get("/health").json()["map_loaded"] is True
    assert c.get("/map").json()["num_keyframes"] == 3


def test_decode_roundtrip():
    data, img = png_bytes()
    back = decode_image(data)
    np.testing.assert_array_equal(back, img)
    data, img = png_bytes((48, 64))
    np.testing.assert_array_equal(decode_image(data), img)


def test_localize_rectified_multipart(client):
    c, fake = client
    data, _ = png_bytes()
    r = c.post(
        "/localize",
        data={"rectified": "true", "camera_label": "camera-rgb", "num_retrieved": "5"},
        files={"image": ("q.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] and body["num_inliers"] == 50
    assert body["T_world_device"]["translation"] == [1, 2, 3]
    kind, shape, cam, label, num = fake.calls[-1]
    assert kind == "image" and shape == (48, 64, 3) and cam.width == 64 and label == "camera-rgb" and num == 5


def test_localize_rectified_with_intrinsics(client):
    c, fake = client
    data, _ = png_bytes()
    calib = {"width": 64, "height": 48, "fx": 10, "fy": 10, "cx": 31.5, "cy": 23.5}
    r = c.post("/localize", data={"rectified": "1", "calibration": json.dumps(calib)}, files={"image": ("q.png", data, "image/png")})
    assert r.status_code == 200
    assert fake.calls[-1][2].fx == 10


def test_localize_raw_json(client):
    c, fake = client
    data, _ = png_bytes()
    r = c.post("/localize/json", json={"image_base64": base64.b64encode(data).decode(), "camera_label": "camera-rgb"})
    assert r.status_code == 200
    assert r.json()["success"] is False and "inliers" in r.json()["message"]
    assert fake.calls[-1][0] == "aria"


def test_bad_requests(client):
    c, _ = client
    data, _ = png_bytes()
    # raw frame without camera label
    r = c.post("/localize", data={}, files={"image": ("q.png", data, "image/png")})
    assert r.status_code == 400
    # rectified without calibration/known label
    r = c.post("/localize", data={"rectified": "true", "camera_label": "camera-x"}, files={"image": ("q.png", data, "image/png")})
    assert r.status_code == 400
    # not an image
    r = c.post("/localize", data={"camera_label": "camera-rgb"}, files={"image": ("q.png", b"nope", "image/png")})
    assert r.status_code == 400
    r = c.post("/localize/json", json={"camera_label": "camera-rgb"})
    assert r.status_code == 400
