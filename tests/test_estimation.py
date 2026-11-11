import numpy as np
import pytest

from objectmodel import synthetic as s
from objectmodel.geometry import angle_between_axes, point_line_distance
from objectmodel.joints import JointFitConfig, fit_joint
from objectmodel.segmentation import segment, segmentation_from_labels
from objectmodel.tree import infer_tree

AXIS_TOL_DEG = 2.0
PIVOT_TOL_M = 0.01


def _majority(gt_labels, mask):
    vals, counts = np.unique(gt_labels[mask], return_counts=True)
    return int(vals[np.argmax(counts)]), counts.max() / counts.sum()


@pytest.mark.parametrize("name", list(s.CATALOGUE))
def test_segmentation_finds_every_part_with_high_purity(tracks, name):
    obj, ts = tracks[name]
    seg = segment(ts)
    assert seg.n_parts == len(obj.parts)
    owners = set()
    for i in range(seg.n_parts):
        owner, purity = _majority(ts.labels, seg.labels == i)
        assert purity > 0.95
        owners.add(owner)
    assert owners == set(range(len(obj.parts)))


@pytest.mark.parametrize("name", list(s.CATALOGUE))
def test_joint_type_axis_and_pivot_match_ground_truth(tracks, name):
    obj, ts = tracks[name]
    seg = segmentation_from_labels(ts, ts.labels)
    for j in obj.joints:
        p, c = obj.part_index(j.parent), obj.part_index(j.child)
        est = fit_joint(seg.poses[p], seg.poses[c], ts.points[0, ts.labels == c])
        assert est.type == j.type
        assert est.confidence > 0.99
        assert angle_between_axes(est.axis, j.axis) < AXIS_TOL_DEG
        if j.type == "revolute":
            assert point_line_distance(est.origin, j.origin, j.axis) < PIVOT_TOL_M
        from objectmodel.benchmark import aligned_states

        states = aligned_states(est.states, est.axis, j.axis)
        assert np.max(np.abs(states - ts.joint_states[j.child])) < 0.03


@pytest.mark.parametrize("name", list(s.CATALOGUE))
def test_tree_topology_matches_ground_truth(tracks, name):
    obj, ts = tracks[name]
    seg = segment(ts)
    tree = infer_tree(ts, seg)
    owner = {i: _majority(ts.labels, seg.labels == i)[0] for i in range(seg.n_parts)}
    got = {(obj.parts[owner[e.parent]].name, obj.parts[owner[e.child]].name) for e in tree.edges}
    assert got == {(j.parent, j.child) for j in obj.joints}


def test_static_child_is_classified_fixed():
    obj = s.laptop()
    frozen = s.SyntheticObject("frozen", obj.parts, [s.GTJoint("base", "lid", "revolute", (1, 0, 0), (0, 0.106, 0.015), 0, 0)])
    ts = s.generate_tracks(frozen, seed=1)
    seg = segmentation_from_labels(ts, ts.labels)
    est = fit_joint(seg.poses[0], seg.poses[1], ts.points[0, ts.labels == 1])
    assert est.type == "fixed"


def test_tiny_rotation_reports_large_uncertainty():
    obj = s.laptop()
    nudge = s.SyntheticObject("nudge", obj.parts, [s.GTJoint("base", "lid", "revolute", (1, 0, 0), (0, 0.106, 0.015), 0, 0.06)])
    ts = s.generate_tracks(nudge, seed=2, noise=0.002)
    seg = segmentation_from_labels(ts, ts.labels)
    big = fit_joint(seg.poses[0], seg.poses[1], ts.points[0, ts.labels == 1], JointFitConfig(bootstrap=40))
    _, ts2 = s.laptop(), s.generate_tracks(s.laptop(), seed=2, noise=0.002)
    seg2 = segmentation_from_labels(ts2, ts2.labels)
    ref = fit_joint(seg2.poses[0], seg2.poses[1], ts2.points[0, ts2.labels == 1], JointFitConfig(bootstrap=40))
    assert big.axis_std_deg > ref.axis_std_deg


def test_segmentation_survives_outlier_tracks():
    ts = s.generate_tracks(s.drawer_cabinet(), seed=5, outlier_frac=0.05)
    seg = segment(ts)
    assert seg.n_parts == 2
    outliers = ts.labels == -1
    assert np.mean(seg.labels[outliers] == -1) > 0.8


def test_segmentation_rejects_static_noise_only_scene():
    ts = s.TrackSet(np.zeros((5, 3, 3)), np.ones((5, 3), bool))
    with pytest.raises(ValueError):
        segment(ts)


def test_trackset_validates_shapes():
    with pytest.raises(ValueError):
        s.TrackSet(np.zeros((5, 3)), np.ones((5, 3), bool))
    with pytest.raises(ValueError):
        s.TrackSet(np.full((2, 2, 3), np.nan), np.ones((2, 2), bool))


@pytest.mark.parametrize("name", list(s.CATALOGUE))
def test_handheld_camera_motion_recovers_tree_and_validates(name):
    from objectmodel.pipeline import run

    obj = s.CATALOGUE[name]()
    ts = s.generate_tracks(obj, seed=3, camera_motion=0.3)
    res = run(ts, use_llm=False)
    seg = res.segmentation
    owner = {i: _majority(ts.labels, seg.labels == i)[0] for i in range(seg.n_parts)}
    got = {(obj.parts[owner[e.parent]].name, obj.parts[owner[e.child]].name, e.joint.type) for e in res.tree.edges}
    assert got == {(j.parent, j.child, j.type) for j in obj.joints}
    assert res.validation.passed, res.validation.to_dict()
