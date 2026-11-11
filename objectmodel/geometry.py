"""Rigid-body geometry primitives: Kabsch fitting, rotations, poses.

Poses are (R, t) pairs mapping reference-frame points x to R @ x + t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Pose:
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3), np.zeros(3))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R.T + self.t

    def inverse(self) -> "Pose":
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def compose(self, other: "Pose") -> "Pose":
        """self ∘ other (apply other first)."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> Pose:
    """Weighted least-squares rigid transform with R @ P_i + t ≈ Q_i."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"kabsch expects matching (N,3) arrays, got {P.shape} and {Q.shape}")
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    if w.sum() <= EPS:
        raise ValueError("kabsch needs positive total weight")
    w = w / w.sum()
    cp = w @ P
    cq = w @ Q
    H = (P - cp).T @ ((Q - cq) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Pose(R, cq - R @ cp)


def batch_kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised weighted Kabsch over a batch: P,Q (B,N,3), w (B,N) -> R (B,3,3), t (B,3)."""
    w = np.ones(P.shape[:2]) if w is None else np.asarray(w, float)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), EPS)
    cp = np.einsum("bn,bni->bi", w, P)[:, None, :]
    cq = np.einsum("bn,bni->bi", w, Q)[:, None, :]
    H = np.einsum("bni,bnj->bij", (P - cp) * w[..., None], Q - cq)
    U, _, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, 1, 2)
    d = np.sign(np.linalg.det(V @ np.swapaxes(U, 1, 2)))
    d[d == 0] = 1.0
    D = np.zeros_like(H)
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = V @ D @ np.swapaxes(U, 1, 2)
    t = cq[:, 0, :] - np.einsum("bij,bj->bi", R, cp[:, 0, :])
    return R, t


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < EPS:
        return np.eye(3)
    k = skew(axis / n)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rotation_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def signed_angle_about(R: np.ndarray, axis: np.ndarray) -> float:
    """Rotation angle of R about a known unit axis (signed, in (-pi, pi])."""
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = 0.5 * float(np.dot(axis, vee))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    return float(np.arctan2(s, c))


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw so that R = Rz(y) Ry(p) Rx(r)."""
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    Rx = axis_angle_to_matrix([1, 0, 0], r)
    Ry = axis_angle_to_matrix([0, 1, 0], p)
    Rz = axis_angle_to_matrix([0, 0, 1], y)
    return Rz @ Ry @ Rx


def angle_between_axes(a: np.ndarray, b: np.ndarray, directed: bool = False) -> float:
    """Angle in degrees between two axes (sign-agnostic unless directed)."""
    a = np.asarray(a, float) / max(np.linalg.norm(a), EPS)
    b = np.asarray(b, float) / max(np.linalg.norm(b), EPS)
    c = float(np.dot(a, b))
    if not directed:
        c = abs(c)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def point_line_distance(p: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, float) / max(np.linalg.norm(axis), EPS)
    d = np.asarray(p, float) - np.asarray(origin, float)
    return float(np.linalg.norm(d - np.dot(d, axis) * axis))


def _box_for_rotation(pts: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    local = pts @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    ext = hi - lo
    return float(np.prod(np.maximum(ext, 1e-4))), R @ ((lo + hi) / 2), ext


def oriented_bounding_box(pts: np.ndarray, yaw_steps: int = 36) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate minimum-volume box -> (center, R columns = box axes, full extents).

    Candidates: world axes, PCA axes, and world axes swept in yaw (gravity-aligned
    furniture). PCA alone is unstable for near-cubic shapes.
    """
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    if len(pts) < 4:
        return c, np.eye(3), np.full(3, 0.01)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    pca = Vt.T.copy()
    if np.linalg.det(pca) < 0:
        pca[:, 2] *= -1
    cands = [np.eye(3), pca]
    for k in range(1, yaw_steps):
        cands.append(axis_angle_to_matrix([0, 0, 1], np.pi / 2 * k / yaw_steps))
    best = min((_box_for_rotation(pts, R) + (R,) for R in cands), key=lambda x: x[0])
    _, center, ext, R = best
    return center, R, ext
