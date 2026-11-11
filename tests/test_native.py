"""C++ core parity with the NumPy implementations (skipped when the library isn't built)."""
import numpy as np
import pytest

from objectmodel import native

pytestmark = pytest.mark.skipif(not native.available(), reason="build with: cmake -S cpp -B cpp/build && cmake --build cpp/build")


@pytest.fixture(scope="module")
def laptop_motions():
    from objectmodel import synthetic as s
    from objectmodel.joints import relative_motions
    from objectmodel.segmentation import segmentation_from_labels
    ts = s.generate_tracks(s.laptop(), seed=0)
    seg = segmentation_from_labels(ts, ts.labels)
    return relative_motions(seg.poses[0], seg.poses[1]), ts.points[0, ts.labels == 1].mean(axis=0)


def test_kabsch_matches_numpy():
    from objectmodel.geometry import axis_angle_to_matrix, kabsch
    rng = np.random.default_rng(1)
    P = rng.normal(size=(40, 3))
    Q = P @ axis_angle_to_matrix([0.3, -0.2, 0.9], 1.1).T + [0.4, 0.1, -0.2] + rng.normal(0, 1e-3, (40, 3))
    w = rng.random(40)
    ref = kabsch(P, Q, w)
    R, t = native.kabsch(P, Q, w)
    assert np.allclose(R, ref.R, atol=1e-12) and np.allclose(t, ref.t, atol=1e-12)


def test_revolute_fit_matches_numpy(laptop_motions):
    from objectmodel.joints import _fit_revolute
    motions, c = laptop_motions
    u, p, th = _fit_revolute(motions, c)
    u2, p2, th2 = native.fit_revolute(motions, c)
    assert np.allclose(u, u2, atol=1e-10) and np.allclose(p, p2, atol=1e-10) and np.allclose(th, th2, atol=1e-10)


def test_prismatic_fit_matches_numpy():
    from objectmodel import synthetic as s
    from objectmodel.joints import _fit_prismatic, relative_motions
    from objectmodel.segmentation import segmentation_from_labels
    ts = s.generate_tracks(s.drawer_cabinet(), seed=0)
    seg = segmentation_from_labels(ts, ts.labels)
    motions = relative_motions(seg.poses[0], seg.poses[1])
    u, d = _fit_prismatic(motions)
    u2, d2 = native.fit_prismatic(motions)
    assert np.allclose(u, u2, atol=1e-10) and np.allclose(d, d2, atol=1e-10)


def test_dls_step_matches_numpy():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(3, 9))
    e = rng.normal(size=3)
    ref = J.T @ np.linalg.solve(J @ J.T + 0.03 ** 2 * np.eye(3), e)
    assert np.allclose(native.dls_step(J, e, 0.03), ref, atol=1e-12)


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        native.kabsch(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        native.dls_step(np.zeros((2, 4)), np.zeros(3), 0.1)


def test_wrong_buffer_lengths_are_rejected():
    P = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError):
        native.kabsch(P, P + 1, w=np.ones(3))
    with pytest.raises(ValueError):
        native.dls_step(np.zeros((3, 4)), np.zeros(2), 0.1)
