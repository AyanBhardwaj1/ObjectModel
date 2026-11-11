"""Rigid motion segmentation of 3D point tracks.

Sequential RANSAC over rigid-motion hypotheses: sample a small spatially local
set of points, fit a per-frame rigid transform with Kabsch, score all points by
their worst-case residual over time, keep the largest consensus set, repeat.
Points near a hinge are rigid with both neighbours; final assignment picks the
best-fitting part, which is harmless for joint estimation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose, batch_kabsch, kabsch
from .synthetic import TrackSet


@dataclass(frozen=True)
class SegmentationConfig:
    inlier_tol: float = 0.006  # metres, per-frame residual tolerance
    min_part_points: int = 25
    max_parts: int = 8
    hypotheses: int = 160
    sample_size: int = 6
    seed: int = 0


RELAX_FACTORS = (1.0, 1.5, 2.0, 3.0)
MIN_SEED_VISIBILITY = 0.8
MIN_LABEL_POINTS = 6


@dataclass
class Segmentation:
    labels: np.ndarray  # (N,) part id, -1 = outlier
    poses: list[list[Pose]]  # [part][frame] mapping frame-0 coords -> frame t
    residual_rms: np.ndarray  # (n_parts,)
    notes: list[str] = field(default_factory=list)

    @property
    def n_parts(self) -> int:
        return len(self.poses)


def estimate_noise(ts: TrackSet) -> float:
    """Per-axis track noise σ (metres) from second temporal differences.

    For smooth motion sampled densely, x[t+1] - 2x[t] + x[t-1] is dominated by
    measurement noise with std σ·√6. The median over tracks is robust to outliers.
    """
    if ts.n_frames < 3:
        return 0.0
    v = ts.visible[2:] & ts.visible[1:-1] & ts.visible[:-2]
    dd = ts.points[2:] - 2 * ts.points[1:-1] + ts.points[:-2]
    # Static tracks from a real tracker can be exactly constant; measure noise on
    # tracks that move when there are enough of them.
    disp = np.where(ts.visible, np.linalg.norm(ts.points - ts.points[0], axis=2), 0).max(axis=0)
    moving = disp > 1e-3
    if moving.sum() >= 10:
        v = v & moving[None, :]
    vals = np.abs(dd[v]).ravel()
    if vals.size == 0:
        return 0.0
    return float(np.median(vals) / 0.6745 / np.sqrt(6))


def adaptive_config(ts: TrackSet, base: "SegmentationConfig" = None) -> "SegmentationConfig":
    from dataclasses import replace

    base = base or SegmentationConfig()
    sigma = estimate_noise(ts)
    return replace(base, inlier_tol=max(base.inlier_tol, 4.0 * sigma))


def _point_residuals(ts: TrackSet, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Residual (T,N) of every point under per-frame transforms R (T,3,3), t (T,3)."""
    ref = ts.points[0]
    pred = np.einsum("tij,nj->tni", R, ref) + t[:, None, :]
    res = np.linalg.norm(pred - ts.points, axis=2)
    return np.where(ts.visible, res, 0.0)


def _robust_point_score(res: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Per-point ~90th-percentile residual over visible frames (robust to blips).

    Invisible frames count as zero residual; np.partition keeps this O(T) per point
    (nanpercentile is ~10x slower and this runs for every RANSAC hypothesis).
    """
    T = res.shape[0]
    filled = np.where(vis, res, 0.0)
    k = min(T - 1, int(np.ceil(0.9 * T)) - 1)
    return np.partition(filled, k, axis=0)[k]


def fit_part_poses(ts: TrackSet, idx: np.ndarray) -> list[Pose]:
    ref = ts.points[0, idx]
    poses = []
    for f in range(ts.n_frames):
        v = ts.visible[f, idx]
        if v.sum() < 3:
            poses.append(poses[-1] if poses else Pose.identity())
            continue
        poses.append(kabsch(ref[v], ts.points[f, idx][v]))
    return poses


def segment(ts: TrackSet, cfg: SegmentationConfig = SegmentationConfig()) -> Segmentation:
    rng = np.random.default_rng(cfg.seed)
    N = ts.n_points
    # Hypothesis seeds must be visible in most frames; per-frame gaps are handled
    # by visibility-weighted Kabsch.
    mostly = ts.visible.mean(axis=0) >= MIN_SEED_VISIBILITY
    ref = ts.points[0]
    unassigned = np.ones(N, bool)
    parts: list[np.ndarray] = []
    tols: list[float] = []

    while unassigned.sum() >= cfg.min_part_points and len(parts) < cfg.max_parts:
        cand = np.flatnonzero(unassigned & mostly)
        if len(cand) < cfg.sample_size:
            break
        seeds = rng.choice(cand, size=min(cfg.hypotheses, len(cand)), replace=len(cand) < cfg.hypotheses)
        d = np.linalg.norm(ref[cand][None, :, :] - ref[seeds][:, None, :], axis=2)
        k = cfg.sample_size
        nn = cand[np.argsort(d, axis=1)[:, :k]]  # (H,k) local neighbourhoods
        H_, T_ = len(seeds), ts.n_frames
        P = np.broadcast_to(ref[nn][:, None], (H_, T_, k, 3)).reshape(-1, k, 3)
        Q = np.transpose(ts.points[:, nn], (1, 0, 2, 3)).reshape(-1, k, 3)
        W = np.transpose(ts.visible[:, nn], (1, 0, 2)).reshape(-1, k).astype(float)
        R, t = batch_kabsch(P, Q, W)
        R = R.reshape(H_, T_, 3, 3)
        t = t.reshape(H_, T_, 3)
        best_count, best_inl, tol = -1, None, cfg.inlier_tol
        scores = [_robust_point_score(_point_residuals(ts, R[h], t[h]), ts.visible) for h in range(len(seeds))]
        # Progressive relaxation: strict tolerance first; looser only for what remains
        # (real trackers drift more on fast-moving parts than on the static base).
        for tol in (cfg.inlier_tol * f for f in RELAX_FACTORS):
            for score in scores:
                inl = (score < tol) & unassigned
                c = int(inl.sum())
                if c > best_count:
                    best_count, best_inl = c, inl
            if best_count >= cfg.min_part_points:
                break
        if best_count < cfg.min_part_points:
            break
        # refine: refit on the consensus set, re-collect inliers
        inl = best_inl
        for _ in range(2):
            poses = fit_part_poses(ts, np.flatnonzero(inl))
            Rr = np.stack([p.R for p in poses])
            tr = np.stack([p.t for p in poses])
            score = _robust_point_score(_point_residuals(ts, Rr, tr), ts.visible)
            inl = (score < tol) & unassigned
        if inl.sum() < cfg.min_part_points:
            break
        parts.append(np.flatnonzero(inl))
        tols.append(tol)
        unassigned &= ~inl

    if not parts:
        raise ValueError("segmentation found no rigid part; check units (metres) and noise level")

    # final assignment: each point to its best-fitting part
    part_poses = [fit_part_poses(ts, idx) for idx in parts]
    scores = np.stack([
        _robust_point_score(_point_residuals(ts, np.stack([p.R for p in pp]), np.stack([p.t for p in pp])), ts.visible)
        for pp in part_poses
    ])
    labels = np.argmin(scores / np.asarray(tols)[:, None], axis=0)
    labels[scores[labels, np.arange(N)] > 2 * np.asarray(tols)[labels]] = -1
    # drop parts that lost their support, then refit
    keep = [i for i in range(len(parts)) if (labels == i).sum() >= cfg.min_part_points]
    remap = {old: new for new, old in enumerate(keep)}
    labels = np.array([remap.get(int(l), -1) for l in labels])
    poses = [fit_part_poses(ts, np.flatnonzero(labels == i)) for i in range(len(keep))]
    rms = np.array([_part_rms(ts, np.flatnonzero(labels == i), poses[i]) for i in range(len(keep))])
    return Segmentation(labels, poses, rms)


def _part_rms(ts: TrackSet, idx: np.ndarray, poses: list[Pose]) -> float:
    R = np.stack([p.R for p in poses])
    t = np.stack([p.t for p in poses])
    sub = TrackSet(ts.points[:, idx], ts.visible[:, idx])
    res = _point_residuals(sub, R, t)
    return float(np.sqrt((res[sub.visible] ** 2).mean()))


def segmentation_from_labels(ts: TrackSet, labels: np.ndarray) -> Segmentation:
    """Use externally supplied part labels (e.g. SAM masks) instead of RANSAC."""
    labels = np.asarray(labels, int).copy()
    if labels.shape != (ts.n_points,):
        raise ValueError("labels must have one entry per track")
    notes = []
    for i in np.unique(labels[labels >= 0]):
        n = int((labels == i).sum())
        if n < MIN_LABEL_POINTS:
            labels[labels == i] = -1
            notes.append(f"ignored part label {int(i)}: only {n} tracks (need {MIN_LABEL_POINTS})")
    ids = sorted(int(i) for i in np.unique(labels) if i >= 0)
    remap = {old: new for new, old in enumerate(ids)}
    lab = np.array([remap.get(int(l), -1) for l in labels])
    poses = [fit_part_poses(ts, np.flatnonzero(lab == i)) for i in range(len(ids))]
    rms = np.array([_part_rms(ts, np.flatnonzero(lab == i), poses[i]) for i in range(len(ids))])
    return Segmentation(lab, poses, rms, notes)
