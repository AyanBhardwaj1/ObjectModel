"""Joint estimation between two rigid parts from their per-frame poses.

Given parent poses P_t and child poses C_t (frame-0 coords -> frame t), the
child's motion in the parent's frame-0 coordinates is M_t = P_t^-1 ∘ C_t.
We fit three models to {M_t} and pick one by BIC (Sturm et al., JAIR 2011):

* fixed      M_t = I
* prismatic  M_t = (I, d_t u)                        u = top singular vector of {t_t}
* revolute   M_t = rotation by θ_t about line (u, p) u = null vector of stacked (R_t - I),
                                                      p solves (I - R_t) p = t_t
Uncertainty comes from a frame bootstrap of the chosen model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import Pose, angle_between_axes, axis_angle_to_matrix, rotation_angle, signed_angle_about

JOINT_TYPES = ("fixed", "prismatic", "revolute")
MIN_BOOTSTRAP_FRAMES = 6


@dataclass(frozen=True)
class JointFitConfig:
    noise_floor: float = 0.0015  # metres; variance floor for BIC so perfect fits don't dominate
    min_motion: float = 0.004  # metres; below this the child is considered fixed
    min_rotation_deg: float = 2.0  # below this a revolute is indistinguishable from prismatic
    limit_margin: float = 0.0  # fraction of observed range added to each limit
    bootstrap: int = 30
    seed: int = 0


@dataclass
class JointEstimate:
    type: str
    axis: np.ndarray  # unit vector, parent frame-0 coords
    origin: np.ndarray  # point on axis (revolute) / child centroid (prismatic)
    states: np.ndarray  # (T,) joint values, 0 at frame 0
    lower: float
    upper: float
    rms: float  # metres, point residual of the chosen model
    confidence: float  # posterior probability of the chosen type
    type_probs: dict = field(default_factory=dict)
    axis_std_deg: float = 0.0
    origin_std: float = 0.0
    candidates: dict = field(default_factory=dict)  # type -> rms

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("axis", "origin", "states"):
            d[k] = np.asarray(d[k]).round(6).tolist()
        return d


def relative_motions(parent: list[Pose], child: list[Pose]) -> list[Pose]:
    return [p.inverse().compose(c) for p, c in zip(parent, child)]


def _rms(pts: np.ndarray, motions: list[Pose], model: list[Pose]) -> float:
    err = [np.linalg.norm(m.apply(pts) - h.apply(pts), axis=1) for m, h in zip(motions, model)]
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def _fit_prismatic(motions: list[Pose]) -> tuple[np.ndarray, np.ndarray]:
    T = np.stack([m.t for m in motions])
    _, _, Vt = np.linalg.svd(T, full_matrices=False)
    u = Vt[0]
    d = T @ u
    if d[np.argmax(np.abs(d))] < 0:  # orient so the dominant motion is positive
        u, d = -u, -d
    return u, d


def _fit_revolute(motions: list[Pose], centroid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.concatenate([m.R - np.eye(3) for m in motions])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)  # economy SVD: the full U would be 3T x 3T
    u = Vt[-1]
    M = np.concatenate([np.eye(3) - m.R for m in motions])
    b = np.concatenate([m.t for m in motions])
    p, *_ = np.linalg.lstsq(M, b, rcond=None)
    p = p + np.dot(centroid - p, u) * u  # nearest axis point to the part
    theta = np.array([signed_angle_about(m.R, u) for m in motions])
    if theta[np.argmax(np.abs(theta))] < 0:
        u, theta = -u, -theta
    theta = np.unwrap(theta)
    return u, p, theta


def _revolute_model(u, p, theta) -> list[Pose]:
    out = []
    for th in theta:
        R = axis_angle_to_matrix(u, th)
        out.append(Pose(R, p - R @ p))
    return out


def _bic(rms: float, n_obs: int, k: int, floor: float) -> float:
    var = max(rms, floor) ** 2
    return n_obs * np.log(var) + k * np.log(n_obs)


def fit_joint(
    parent: list[Pose],
    child: list[Pose],
    child_points: np.ndarray,
    cfg: JointFitConfig = JointFitConfig(),
) -> JointEstimate:
    """Fit and select a joint for child relative to parent.

    child_points are the child's points in frame-0 coordinates (used for residuals).
    """
    motions = relative_motions(parent, child)
    pts = np.asarray(child_points, float)
    if len(pts) > 200:
        pts = pts[np.linspace(0, len(pts) - 1, 200).astype(int)]
    centroid = pts.mean(axis=0)
    T = len(motions)
    n_obs = 3 * len(pts) * T
    ident = [Pose.identity()] * T

    rms_fixed = _rms(pts, motions, ident)
    u_p, d = _fit_prismatic(motions)
    rms_pris = _rms(pts, motions, [Pose(np.eye(3), di * u_p) for di in d])
    u_r, p_r, th = _fit_revolute(motions, centroid)
    rms_rev = _rms(pts, motions, _revolute_model(u_r, p_r, th))
    max_rot = max(np.degrees(rotation_angle(m.R)) for m in motions)

    floor = cfg.noise_floor
    bic = {
        "fixed": _bic(rms_fixed, n_obs, 0, floor),
        "prismatic": _bic(rms_pris, n_obs, 2 + T, floor),
        "revolute": _bic(rms_rev, n_obs, 4 + T, floor),
    }
    # Hard gates that BIC alone handles poorly with tiny motions.
    max_disp = max(float(np.max(np.linalg.norm(m.apply(pts) - pts, axis=1))) for m in motions)
    if max_disp < cfg.min_motion:
        bic["prismatic"] = bic["revolute"] = np.inf
    if max_rot < cfg.min_rotation_deg:
        bic["revolute"] = np.inf
    probs = _bic_posterior(bic)
    kind = max(probs, key=probs.get)

    if kind == "revolute":
        axis, origin, states, rms = u_r, p_r, th, rms_rev
    elif kind == "prismatic":
        axis, origin, states, rms = u_p, centroid, d, rms_pris
    else:
        axis, origin, states, rms = np.array([0.0, 0.0, 1.0]), centroid, np.zeros(T), rms_fixed

    lo, hi = float(states.min()), float(states.max())
    margin = cfg.limit_margin * (hi - lo)
    est = JointEstimate(
        type=kind, axis=axis, origin=origin, states=states,
        lower=min(0.0, lo - margin), upper=max(0.0, hi + margin),
        rms=rms, confidence=probs[kind], type_probs=probs,
        candidates={"fixed": rms_fixed, "prismatic": rms_pris, "revolute": rms_rev},
    )
    if kind != "fixed":
        if cfg.bootstrap > 0 and T >= MIN_BOOTSTRAP_FRAMES:
            est.axis_std_deg, est.origin_std = _bootstrap(motions, centroid, kind, axis, origin, cfg)
        else:
            est.axis_std_deg = est.origin_std = float("nan")  # unknown, not zero
    return est


def _bic_posterior(bic: dict) -> dict:
    vals = np.array([bic[k] for k in JOINT_TYPES], float)
    finite = np.isfinite(vals)
    out = np.zeros_like(vals)
    if finite.any():
        v = -0.5 * (vals[finite] - vals[finite].min())
        w = np.exp(np.clip(v, -700, 0))
        out[finite] = w / w.sum()
    return {k: float(p) for k, p in zip(JOINT_TYPES, out)}


def _bootstrap(motions, centroid, kind, axis, origin, cfg) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed)
    T = len(motions)
    angs, offs = [], []
    for _ in range(cfg.bootstrap):
        idx = rng.choice(T, T, replace=True)
        sub = [motions[i] for i in idx]
        if kind == "revolute":
            u, p, _ = _fit_revolute(sub, centroid)
            d = p - origin
            offs.append(np.linalg.norm(d - np.dot(d, axis) * axis))
        else:
            u, _ = _fit_prismatic(sub)
            offs.append(0.0)
        angs.append(angle_between_axes(u, axis))
    return float(np.sqrt(np.mean(np.square(angs)))), float(np.sqrt(np.mean(np.square(offs))))
