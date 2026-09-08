import numpy as np

from aria_hloc.aria.calibration import PinholeCamera, RectifiedCamera
from aria_hloc.geometry import make_transform, quat_wxyz_to_rotmat
from aria_hloc.map_format import Keyframe, MapMetadata, MapPaths, load_keyframes, save_keyframes


def make_cameras():
    T = make_transform(quat_wxyz_to_rotmat([0.9, 0.1, 0.2, 0.3]), [0.01, -0.02, 0.03])
    return {
        "camera-rgb": RectifiedCamera("camera-rgb", PinholeCamera(1408, 1408, 600, 600, 703.5, 703.5), T, True, {"model": "FISHEYE624"}),
        "camera-slam-front-left": RectifiedCamera("camera-slam-front-left", PinholeCamera(480, 640, 240, 240, 239.5, 319.5), np.eye(4), True),
    }


def test_metadata_roundtrip(tmp_path):
    meta = MapMetadata(make_cameras(), source={"vrs": "a.vrs"}, stats={"n": 1})
    meta.save(tmp_path)
    back = MapMetadata.load(tmp_path)
    assert set(back.cameras) == set(meta.cameras)
    cam = back.cameras["camera-rgb"]
    assert cam.pinhole == meta.cameras["camera-rgb"].pinhole
    np.testing.assert_allclose(cam.T_device_camera, meta.cameras["camera-rgb"].T_device_camera, atol=1e-12)
    assert cam.upright and cam.source == {"model": "FISHEYE624"}
    assert back.feature_conf == "superpoint_aachen"
    assert back.source["vrs"] == "a.vrs"
    assert not MapPaths(tmp_path).is_complete()


def test_keyframes_roundtrip(tmp_path):
    T = make_transform(quat_wxyz_to_rotmat([0.5, 0.5, 0.5, 0.5]), [1, 2, 3])
    kfs = [Keyframe("camera-rgb/1.jpg", "camera-rgb", 1, T, np.eye(4)), Keyframe("camera-rgb/2.jpg", "camera-rgb", 2, T)]
    save_keyframes(kfs, tmp_path)
    back = load_keyframes(tmp_path)
    assert [k.name for k in back] == ["camera-rgb/1.jpg", "camera-rgb/2.jpg"]
    np.testing.assert_allclose(back[0].T_world_camera, T, atol=1e-12)
    np.testing.assert_allclose(back[0].T_world_device, np.eye(4))
    assert back[1].T_world_device is None


def test_pinhole_scaled():
    cam = PinholeCamera(640, 480, 500, 500, 319.5, 239.5).scaled(0.5)
    assert (cam.width, cam.height) == (320, 240)
    assert cam.fx == 250 and cam.cx == 159.5
