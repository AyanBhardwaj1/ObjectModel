import numpy as np
import pytest

from objectmodel.geometry import (
    Pose, angle_between_axes, axis_angle_to_matrix, batch_kabsch, kabsch, matrix_to_quat_wxyz,
    matrix_to_rpy, oriented_bounding_box, rpy_to_matrix, signed_angle_about,
)

rng = np.random.default_rng(0)


def random_rotation():
    return axis_angle_to_matrix(rng.normal(size=3), rng.uniform(-np.pi, np.pi))


def test_kabsch_recovers_exact_transform():
    P = rng.normal(size=(50, 3))
    R, t = random_rotation(), rng.normal(size=3)
    est = kabsch(P, P @ R.T + t)
    assert np.allclose(est.R, R, atol=1e-9) and np.allclose(est.t, t, atol=1e-9)


def test_kabsch_never_returns_reflection():
    P = rng.normal(size=(20, 3))
    Q = P * np.array([1, 1, -1])  # mirrored
    assert np.linalg.det(kabsch(P, Q).R) == pytest.approx(1.0)


def test_kabsch_rejects_bad_shapes():
    with pytest.raises(ValueError):
        kabsch(np.zeros((4, 3)), np.zeros((5, 3)))


def test_batch_kabsch_matches_single():
    P = rng.normal(size=(4, 10, 3))
    Rs = np.stack([random_rotation() for _ in range(4)])
    Q = np.einsum("bij,bnj->bni", Rs, P) + 0.3
    R, t = batch_kabsch(P, Q)
    assert np.allclose(R, Rs, atol=1e-9) and np.allclose(t, 0.3, atol=1e-9)


def test_pose_inverse_and_compose():
    a = Pose(random_rotation(), rng.normal(size=3))
    ident = a.compose(a.inverse())
    assert np.allclose(ident.R, np.eye(3)) and np.allclose(ident.t, 0)


@pytest.mark.parametrize("angle", [-2.5, -0.3, 0.0, 0.7, 3.0])
def test_signed_angle_about_axis(angle):
    axis = np.array([0.0, 0.6, 0.8])
    assert signed_angle_about(axis_angle_to_matrix(axis, angle), axis) == pytest.approx(angle, abs=1e-9)


def test_rpy_roundtrip_and_quat_unit():
    R = random_rotation()
    assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R, atol=1e-9)
    assert np.linalg.norm(matrix_to_quat_wxyz(R)) == pytest.approx(1.0)


def test_axis_angle_is_sign_agnostic_by_default():
    assert angle_between_axes([0, 0, 1], [0, 0, -1]) == pytest.approx(0.0)
    assert angle_between_axes([0, 0, 1], [0, 0, -1], directed=True) == pytest.approx(180.0)


def test_obb_handles_near_cubic_box():
    pts = (rng.random((2000, 3)) - 0.5) * [0.6, 0.5, 0.8]
    _, _, size = oriented_bounding_box(pts)
    assert np.allclose(sorted(size), sorted([0.6, 0.5, 0.8]), atol=0.02)
