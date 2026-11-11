import numpy as np
import pytest

pytest.importorskip("cv2")

from objectmodel.geometry import angle_between_axes, point_line_distance  # noqa: E402
from objectmodel.pipeline import run  # noqa: E402
from objectmodel.video import VideoConfig, lift_depth, render_synthetic_video, tracks_from_video  # noqa: E402


@pytest.fixture(scope="module")
def laptop_video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("vid") / "laptop.avi")
    return path, render_synthetic_video(path, "laptop")


def test_video_laptop_recovers_hinge(laptop_video):
    path, gt = laptop_video
    ts = tracks_from_video(path, VideoConfig(scale_mm=300))
    res = run(ts, hint="laptop", use_llm=False)
    assert len(res.model.joints) == 1
    j = res.model.joints[0]
    assert j.type == "revolute"
    # planar capture: hinge axis is the viewing direction (world y)
    assert angle_between_axes(j.axis, [0, 1, 0]) < 3.0
    assert abs(j.upper - gt["range_rad"]) < 0.06
    # pivot sits on the rendered hinge pixel (scale-consistent check)
    m_per_px = j.origin[0] / gt["pivot_px"][0]
    assert -j.origin[2] / gt["pivot_px"][1] == pytest.approx(m_per_px, rel=0.03)
    assert res.validation.passed


def test_video_drawer_is_prismatic(tmp_path):
    path = str(tmp_path / "drawer.avi")
    render_synthetic_video(path, "drawer")
    res = run(tracks_from_video(path, VideoConfig(scale_mm=300)), hint="drawer", use_llm=False)
    j = res.model.joints[0]
    assert j.type == "prismatic" and angle_between_axes(j.axis, [1, 0, 0]) < 3.0


def test_missing_video_raises():
    with pytest.raises(FileNotFoundError):
        tracks_from_video("/nonexistent/clip.mp4")


def test_depth_lifting_backprojects_center_pixel():
    depth = np.full((1, 100, 200), 2.0)
    uv = np.array([[[100.0, 50.0], [200.0, 50.0]]])
    pts, vis = lift_depth(uv, np.ones((1, 2), bool), depth, fov_deg=90)
    assert np.allclose(pts[0, 0], [0, 2.0, 0])
    assert vis.all()
    assert pts[0, 1, 0] == pytest.approx(2.0, rel=0.02)  # 90° FOV: image edge at x = z


def test_depth_lifting_masks_invalid_depth():
    depth = np.zeros((1, 10, 10))
    _, vis = lift_depth(np.zeros((1, 1, 2)), np.ones((1, 1), bool), depth, 60)
    assert not vis.any()
