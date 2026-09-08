import numpy as np
import pytest

from aria_hloc.aria.calibration import PinholeCamera, RectifiedCamera
from aria_hloc.geometry import invert_transform, make_transform, quat_wxyz_to_rotmat
from aria_hloc.map_format import Keyframe
from aria_hloc.mapping.reference_model import build_colmap_model, write_reference_model
from aria_hloc.utils.read_write_model import read_model


def make_data():
    cams = {"camera-rgb": RectifiedCamera("camera-rgb", PinholeCamera(640, 480, 500, 500, 319.5, 239.5), np.eye(4))}
    kfs = []
    for i in range(3):
        T = make_transform(quat_wxyz_to_rotmat([0.9, 0.1 * i, 0.2, 0.3]), [i, 0.5, 0])
        kfs.append(Keyframe(f"camera-rgb/{i}.jpg", "camera-rgb", i, T))
    return cams, kfs


def test_build_model_poses_are_cam_from_world():
    cams, kfs = make_data()
    c, images, p = build_colmap_model(cams, kfs)
    assert len(c) == 1 and len(images) == 3 and p == {}
    for img_id, img in images.items():
        kf = kfs[img_id - 1]
        T_cam_world = invert_transform(kf.T_world_camera)
        np.testing.assert_allclose(img.qvec2rotmat(), T_cam_world[:3, :3], atol=1e-10)
        np.testing.assert_allclose(img.tvec, T_cam_world[:3, 3], atol=1e-10)
        assert img.name == kf.name
        assert c[img.camera_id].model == "PINHOLE"


def test_unknown_camera():
    cams, kfs = make_data()
    kfs[0].label = "camera-other"
    with pytest.raises(KeyError):
        build_colmap_model(cams, kfs)


@pytest.mark.parametrize("ext", [".bin", ".txt"])
def test_write_and_read_back(tmp_path, ext):
    cams, kfs = make_data()
    write_reference_model(cams, kfs, tmp_path, ext=ext)
    c, images, p = read_model(str(tmp_path), ext=ext)
    assert len(images) == 3
    np.testing.assert_allclose(c[1].params, [500, 500, 319.5, 239.5])
    assert images[2].name == "camera-rgb/1.jpg"


def test_pycolmap_reads_reference(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    cams, kfs = make_data()
    write_reference_model(cams, kfs, tmp_path)
    rec = pycolmap.Reconstruction(str(tmp_path))
    assert len(rec.images) == 3
    img = next(im for im in rec.images.values() if im.name == "camera-rgb/2.jpg")
    cfw = img.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    np.testing.assert_allclose(np.asarray(cfw.matrix()), invert_transform(kfs[2].T_world_camera)[:3], atol=1e-6)
