import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from objectmodel import synthetic as s
from objectmodel.export import to_mjcf, to_urdf
from objectmodel.pipeline import run, write_outputs
from objectmodel.validate import validate
from objectmodel.viewer import render_viewer


@pytest.fixture(scope="module")
def cabinet():
    return run(s.generate_tracks(s.kitchen_cabinet(), seed=21), hint="kitchen cabinet", use_llm=False)


def test_pipeline_produces_sim_validated_model(cabinet):
    v = cabinet.validation
    assert v.compiled and v.stable and v.passed, v.to_dict()
    assert v.replay_overall_mm < 5.0


def test_urdf_is_well_formed_with_required_limits(cabinet):
    root = ET.fromstring(to_urdf(cabinet.model))
    assert root.tag == "robot"
    assert len(root.findall("link")) == 3
    for j in root.findall("joint"):
        if j.get("type") in ("revolute", "prismatic"):
            lim = j.find("limit")
            assert lim is not None and float(lim.get("lower")) < float(lim.get("upper"))
            assert lim.get("effort") and lim.get("velocity")
    for link in root.findall("link"):
        assert float(link.find("inertial/mass").get("value")) > 0


def test_mjcf_uses_radians_and_compiles(cabinet):
    xml = to_mjcf(cabinet.model)
    assert 'angle="radian"' in xml
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.njnt == 2 and m.nbody == 4  # world + 3 links


def test_validator_catches_a_wrong_axis(cabinet):
    model = cabinet.model
    door = next(j for j in model.joints if j.type == "revolute")
    original = door.axis.copy()
    try:
        door.axis = np.array([1.0, 0.0, 0.0])  # hinge pointing the wrong way
        bad = validate(model, cabinet.tracks, cabinet.segmentation.labels)
        assert not bad.passed
        assert bad.replay_overall_mm > 10
    finally:
        door.axis = original


def test_validator_reports_invalid_inertia(cabinet):
    link = cabinet.model.links[1]
    phys = link.physics
    try:
        object.__setattr__(phys, "inertia_diag", np.array([1.0, 1.0, 5.0]))
        r = validate(cabinet.model, cabinet.tracks, cabinet.segmentation.labels)
        assert any("inertia" in e for e in r.errors) and not r.passed
    finally:
        object.__setattr__(phys, "inertia_diag", phys.inertia_diag)


def test_write_outputs_creates_all_artifacts(cabinet, tmp_path):
    files = write_outputs(cabinet, str(tmp_path))
    for path in files.values():
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 0
    scene = json.loads((tmp_path / "model.json").read_text())
    assert scene["validation"]["passed"] is True
    assert {j["type"] for j in scene["joints"]} == {"revolute", "prismatic"}


def test_viewer_escapes_script_breakout():
    html = render_viewer({"name": "x", "notes": ["</script><script>alert(1)</script>"], "links": [], "joints": [], "root": ""})
    assert "</script><script>alert(1)" not in html
    assert html.count("</script>") == 3  # two CDN tags + the inline script
