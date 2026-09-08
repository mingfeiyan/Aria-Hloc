"""End-to-end test on a synthetic scene: build an hloc map from posed renders,
then relocalize held-out renders and compare with the ground-truth poses.

Runs only with ``ARIA_HLOC_E2E=1`` and hloc/torch/pycolmap installed (it downloads
model weights and takes a few minutes on CPU). The retrieval model can be chosen
with ``ARIA_HLOC_E2E_RETRIEVAL`` (default ``netvlad``).
"""

import os
from pathlib import Path

import numpy as np
import pytest

E2E = os.environ.get("ARIA_HLOC_E2E") == "1"
pytestmark = pytest.mark.skipif(not E2E, reason="set ARIA_HLOC_E2E=1 to run the synthetic end-to-end test")

if E2E:
    pytest.importorskip("torch")
    pytest.importorskip("hloc")
    pytest.importorskip("pycolmap")
    import cv2

from aria_hloc.aria.calibration import PinholeCamera, RectifiedCamera  # noqa: E402
from aria_hloc.geometry import invert_transform, make_transform, quat_wxyz_to_rotmat, relative_pose_error  # noqa: E402
from aria_hloc.map_format import Keyframe, MapMetadata, MapPaths, save_keyframes  # noqa: E402

PINHOLE = PinholeCamera(640, 480, 400.0, 400.0, 319.5, 239.5)
TEXTURE_PX = 2400
METERS_PER_PX = 8.0 / TEXTURE_PX  # texture covers 8 x 8 m centred at the origin


def make_texture(rng) -> np.ndarray:
    tex = np.full((TEXTURE_PX, TEXTURE_PX, 3), 200, np.uint8)
    for _ in range(600):
        c = rng.integers(0, 255, 3).tolist()
        x, y = rng.integers(0, TEXTURE_PX, 2)
        if rng.random() < 0.5:
            w, h = rng.integers(10, 120, 2)
            cv2.rectangle(tex, (int(x), int(y)), (int(x + w), int(y + h)), c, -1)
        else:
            cv2.circle(tex, (int(x), int(y)), int(rng.integers(5, 60)), c, -1)
    noise = rng.normal(0, 8, tex.shape)
    return np.clip(tex.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def look_down_pose(x, y, z, roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    """T_world_cam of a camera at (x, y, z) looking down at the z=0 plane."""
    base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)  # z down, x right, y "south"

    def rot(axis, deg):
        a = np.radians(deg) / 2
        q = np.array([np.cos(a)] + [np.sin(a) * v for v in axis])
        return quat_wxyz_to_rotmat(q)

    R = base @ rot([0, 0, 1], yaw_deg) @ rot([1, 0, 0], pitch_deg) @ rot([0, 1, 0], roll_deg)
    return make_transform(R, [x, y, z])


def render(texture: np.ndarray, T_world_cam: np.ndarray, cam: PinholeCamera) -> np.ndarray:
    T_cam_world = invert_transform(T_world_cam)
    R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
    H_img_world = cam.K @ np.column_stack([R[:, 0], R[:, 1], t])  # plane z = 0
    # texture pixel (u, v) -> world (X, Y): X = (u - c) * mpp, Y = (v - c) * mpp
    c = (TEXTURE_PX - 1) / 2.0
    A = np.array([[METERS_PER_PX, 0, -c * METERS_PER_PX], [0, METERS_PER_PX, -c * METERS_PER_PX], [0, 0, 1]])
    H = H_img_world @ A
    return cv2.warpPerspective(texture, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR)


@pytest.fixture(scope="module")
def synthetic_map(tmp_path_factory):
    from aria_hloc.mapping.build_map import MapBuildConfig, run_hloc_pipeline
    from aria_hloc.mapping.reference_model import write_reference_model

    rng = np.random.default_rng(0)
    texture = make_texture(rng)
    map_dir = Path(tmp_path_factory.mktemp("map"))
    paths = MapPaths(map_dir)
    T_device_camera = make_transform(quat_wxyz_to_rotmat([0.95, 0.1, 0.2, 0.15]), [0.05, -0.02, 0.01])
    cameras = {"camera-rgb": RectifiedCamera("camera-rgb", PINHOLE, T_device_camera, upright=True)}
    keyframes = []
    grid = np.linspace(-1.5, 1.5, 6)
    ts = 0
    for x in grid:
        for y in grid:
            T = look_down_pose(x, y, 3.0 + rng.uniform(-0.2, 0.2), *rng.uniform(-5, 5, 3))
            ts += 100_000_000
            name = f"camera-rgb/{ts}.png"
            img = render(texture, T, PINHOLE)
            (paths.images / "camera-rgb").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(paths.images / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            keyframes.append(Keyframe(name, "camera-rgb", ts, T, T @ invert_transform(T_device_camera)))
    save_keyframes(keyframes, map_dir)
    write_reference_model(cameras, keyframes, paths.reference)
    retrieval = os.environ.get("ARIA_HLOC_E2E_RETRIEVAL", "netvlad")
    meta = MapMetadata(cameras, retrieval_conf=retrieval, source={"synthetic": True})
    meta.save(map_dir)
    cfg = MapBuildConfig(vrs=Path("none"), mps=Path("none"), output=map_dir, retrieval_conf=retrieval, num_pairs=10)
    meta.stats["hloc"] = run_hloc_pipeline(cfg, paths)
    meta.save(map_dir)
    return map_dir, texture, T_device_camera


def test_map_statistics(synthetic_map):
    map_dir, _, _ = synthetic_map
    meta = MapMetadata.load(map_dir)
    stats = meta.stats["hloc"]
    assert stats["num_points3D"] > 1000
    assert stats["num_images_with_points"] == 36
    assert stats["mean_reprojection_error_px"] < 2.0
    assert MapPaths(map_dir).is_complete()


def test_relocalize_held_out_views(synthetic_map):
    from aria_hloc.localization.relocalizer import Relocalizer, RelocalizerConfig

    map_dir, texture, T_device_camera = synthetic_map
    reloc = Relocalizer(map_dir, RelocalizerConfig(num_retrieved=8, device="cpu"))
    rng = np.random.default_rng(1)
    errors = []
    for _ in range(6):
        T_gt = look_down_pose(*rng.uniform(-1.2, 1.2, 2), 2.6 + rng.uniform(0, 0.8), *rng.uniform(-8, 8, 3))
        img = render(texture, T_gt, PINHOLE)
        res = reloc.localize_image(img, PINHOLE, "camera-rgb")
        assert res.success, res.message
        dt, dR = relative_pose_error(res.T_world_camera, T_gt)
        errors.append((dt, dR))
        # device pose derived through the stored extrinsics
        np.testing.assert_allclose(res.T_world_device, res.T_world_camera @ invert_transform(T_device_camera), atol=1e-9)
        assert res.num_inliers >= 30
        assert set(res.timings_ms) >= {"retrieval", "features", "matching", "pnp"}
    errors = np.array(errors)
    assert np.all(errors[:, 0] < 0.05), errors
    assert np.all(errors[:, 1] < 1.0), errors


def test_service_roundtrip(synthetic_map):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aria_hloc.localization.relocalizer import Relocalizer, RelocalizerConfig
    from aria_hloc.service.app import create_app
    from aria_hloc.service.client import encode_image

    map_dir, texture, _ = synthetic_map
    reloc = Relocalizer(map_dir, RelocalizerConfig(device="cpu"))
    client = TestClient(create_app(relocalizer=reloc))
    T_gt = look_down_pose(0.3, -0.4, 3.1, 2, -3, 4)
    img = render(texture, T_gt, PINHOLE)
    r = client.post(
        "/localize",
        data={"rectified": "true", "camera_label": "camera-rgb"},
        files={"image": ("q.png", encode_image(img), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"]
    dt, dR = relative_pose_error(np.asarray(body["T_world_camera"]["matrix"]), T_gt)
    assert dt < 0.05 and dR < 1.0
