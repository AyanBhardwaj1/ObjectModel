"""The web API's service layer: input validation and response shape."""
import io
import json

import numpy as np
import pytest

from objectmodel import synthetic as s
from objectmodel.service import MAX_POINTS, BadRequest, health, reconstruct_demo, reconstruct_upload


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def test_demo_returns_files_viewer_and_verdict():
    out = reconstruct_demo({"object": "laptop", "noise_mm": 1})
    assert out["summary"]["sim_validated"] is True
    assert set(out["files"]) == {"laptop.urdf", "laptop.xml", "model.json", "report.md"}
    assert "<!doctype html>" in out["viewer_html"]
    json.loads(out["files"]["model.json"])  # strict JSON


@pytest.mark.parametrize("params,msg", [
    ({"object": "rocket"}, "object must be one of"),
    ({"object": "laptop", "noise_mm": 99}, "noise_mm must be between"),
    ({"object": "laptop", "noise_mm": "loud"}, "noise_mm must be a number"),
    ({"object": "laptop", "camera_motion": float("nan")}, "camera_motion must be between"),
])
def test_demo_rejects_bad_parameters(params, msg):
    with pytest.raises(BadRequest, match=msg):
        reconstruct_demo(params)


def test_upload_round_trip():
    ts = s.generate_tracks(s.laptop(), seed=3)
    out = reconstruct_upload(_npz(points=ts.points, visible=ts.visible), {"hint": "laptop"})
    assert out["summary"]["sim_validated"] and out["summary"]["joints"][0]["type"] == "revolute"


@pytest.mark.parametrize("body,msg", [
    (b"", "empty upload"),
    (b"not an archive", "could not read .npz"),
    (_npz(other=np.zeros(3)), "needs a 'points' array"),
    (_npz(points=np.zeros((5, 30))), "points must be"),
    (_npz(points=np.zeros((2, 30, 3))), "need 3"),
    (_npz(points=np.zeros((5, MAX_POINTS + 1, 3))), "need 3"),
    (_npz(points=np.full((5, 30, 3), np.inf)), "NaN or infinity"),
    (_npz(points=np.zeros((5, 30, 3)), visible=np.ones((4, 30), bool)), "visible must be"),
])
def test_upload_rejects_bad_archives(body, msg):
    with pytest.raises(BadRequest, match=msg):
        reconstruct_upload(body, {})


def test_upload_refuses_pickled_objects():
    buf = io.BytesIO()
    np.save(buf, np.array([{"evil": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(BadRequest):
        reconstruct_upload(_npz(points=np.array([{"a": 1}], dtype=object)), {})


def test_upload_too_large():
    with pytest.raises(BadRequest, match="larger than"):
        reconstruct_upload(b"x" * (4 * 1024 * 1024 + 1), {})


def test_health_lists_objects():
    h = health()
    assert h["ok"] and "dresser" in h["objects"]
