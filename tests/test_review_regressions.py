"""Regression tests for findings from the independent code review."""
import io
import json
import urllib.error
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from objectmodel import synthetic as s
from objectmodel.agent import LLMConfig, _open, annotate, chat_json
from objectmodel.cli import main
from objectmodel.export import to_mjcf, to_scene_json, to_urdf
from objectmodel.geometry import rpy_to_matrix
from objectmodel.joints import fit_joint
from objectmodel.model import safe_name
from objectmodel.pipeline import run
from objectmodel.report import _md_escape
from objectmodel.segmentation import segment, segmentation_from_labels
from objectmodel.tree import infer_tree


def test_tracks_hidden_in_first_frame_do_not_corrupt_the_model():
    ts = s.generate_tracks(s.laptop(), seed=4)
    hide = np.random.default_rng(0).random(ts.n_points) < 0.1
    vis = ts.visible.copy()
    vis[0, hide] = False
    pts = np.where(vis[..., None], ts.points, 0.0)
    res = run(s.TrackSet(pts, vis, ts.labels), hint="laptop", labels=ts.labels, use_llm=False)
    assert res.validation.passed, res.validation.to_dict()
    lid = res.model.link(res.model.joints[0].child)
    assert sorted(lid.geometry.size)[0] < 0.03  # lid stays thin (was 0.11 m with the bug)
    assert any("not visible in the first frame" in n for n in res.model.notes)


def _urdf_world_poses(urdf: str, q: dict) -> dict:
    """Minimal URDF forward kinematics (translations + joint motion) for comparison."""
    root = ET.fromstring(urdf)
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    cache = {}

    def pose(link):
        if link in cache:
            return cache[link]
        j = joints.get(link)
        if j is None:
            cache[link] = (np.eye(3), np.zeros(3))
            return cache[link]
        Rp, tp = pose(j.find("parent").get("link"))
        o = np.array(j.find("origin").get("xyz").split(), float)
        R, t = Rp, tp + Rp @ o
        v = q.get(j.get("name"), 0.0)
        if j.get("type") in ("revolute", "continuous"):
            from objectmodel.geometry import axis_angle_to_matrix
            R = R @ axis_angle_to_matrix(np.array(j.find("axis").get("xyz").split(), float), v)
        elif j.get("type") == "prismatic":
            t = t + R @ (np.array(j.find("axis").get("xyz").split(), float) * v)
        cache[link] = (R, t)
        return cache[link]

    return {l.get("name"): pose(l.get("name")) for l in root.findall("link")}


def test_urdf_and_mjcf_agree_on_world_poses():
    res = run(s.generate_tracks(s.kitchen_cabinet(), seed=1), use_llm=False)
    model = res.model
    m = mujoco.MjModel.from_xml_string(to_mjcf(model))
    d = mujoco.MjData(m)
    q = {j.name: 0.5 * (j.lower + j.upper) for j in model.joints}
    for j in model.joints:
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)]] = q[j.name]
    mujoco.mj_kinematics(m, d)
    urdf = _urdf_world_poses(to_urdf(model), q)
    for link in model.links:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link.name)
        R, t = urdf[link.name]
        assert np.allclose(d.xpos[bid], t, atol=1e-5), link.name
        assert np.allclose(d.xmat[bid].reshape(3, 3), R, atol=1e-5), link.name


@pytest.mark.parametrize("content", [None, 42])
def test_llm_non_text_content_falls_back(content, monkeypatch):
    ts = s.generate_tracks(s.drawer_cabinet(), seed=3)
    seg = segment(ts)
    tree = infer_tree(ts, seg)
    import objectmodel.agent as agent

    def opener(req, timeout):
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        r = io.BytesIO(body)
        r.__enter__ = lambda *a: r
        r.__exit__ = lambda *a: False
        return type("R", (), {"__enter__": lambda self: r, "__exit__": lambda self, *a: False})()

    monkeypatch.setattr(agent, "_open", opener)
    monkeypatch.setattr(agent.chat_json, "__defaults__", (opener,))
    ann = annotate(seg, tree, ts, llm=LLMConfig("https://example.test/v1", "", "m"))
    assert ann.source == "rules" and ann.notes


def test_llm_unexpected_exception_still_falls_back(monkeypatch):
    ts = s.generate_tracks(s.drawer_cabinet(), seed=3)
    seg = segment(ts)
    tree = infer_tree(ts, seg)
    import objectmodel.agent as agent

    def boom(*a, **k):
        raise ConnectionResetError("peer reset")

    monkeypatch.setattr(agent, "chat_json", boom)
    ann = annotate(seg, tree, ts, llm=LLMConfig("https://example.test/v1", "", "m"))
    assert ann.source == "rules" and "ConnectionResetError" in ann.notes[0]


def test_redirects_are_refused():
    from objectmodel.agent import _NoRedirect
    import urllib.request

    h = _NoRedirect()
    req = urllib.request.Request("https://api.example/v1/chat/completions", headers={"Authorization": "Bearer k"})
    with pytest.raises(urllib.error.HTTPError):
        h.redirect_request(req, None, 302, "Found", {}, "http://evil.example/")


def test_tiny_user_labels_are_ignored_not_made_root():
    ts = s.generate_tracks(s.laptop(), seed=1)
    labels = ts.labels.copy()
    labels[5] = 7  # a stray one-track label
    res = run(ts, labels=labels, use_llm=False)
    assert len(res.model.links) == 2
    assert any("ignored part label 7" in n for n in res.model.notes)
    assert res.model.joints[0].type == "revolute"


def test_world_is_a_reserved_name():
    assert safe_name("world", "part") == "part_world"
    assert safe_name("World", "part") == "part_World"


def test_too_few_frames_reports_unknown_uncertainty_not_zero():
    ts = s.generate_tracks(s.laptop(), n_frames=5, seed=0, dropout=0.0)
    seg = segmentation_from_labels(ts, ts.labels)
    est = fit_joint(seg.poses[0], seg.poses[1], ts.points[0, ts.labels == 1])
    assert np.isnan(est.axis_std_deg)


def test_scene_json_is_strict_json_even_with_unknown_uncertainty():
    res = run(s.generate_tracks(s.laptop(), n_frames=5, seed=0, dropout=0.0), use_llm=False)
    text = json.dumps(to_scene_json(res.model), allow_nan=False)
    assert '"axis_std_deg": null' in text
    assert any("uncertainty not estimated" in w for w in res.validation.warnings)


def test_cli_rejects_zero_stride(capsys):
    with pytest.raises(SystemExit):
        main(["video", "x.mp4", "--stride", "0"])


def test_markdown_escaping_of_untrusted_notes():
    out = _md_escape("[click](http://evil) <img src=x> `code`")
    import re
    assert not re.search(r"(?<!\\)[\[\]()<>`]", out)  # every special char is backslash-escaped
