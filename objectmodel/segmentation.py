"""Rigid motion segmentation of 3D point tracks.

Sequential RANSAC over rigid-motion hypotheses: sample a small spatially local
set of points, fit a per-frame rigid transform with Kabsch, score all points by
their worst-case residual over time, keep the largest consensus set, repeat.
Points near a hinge are rigid with both neighbours; final assignment picks the
best-fitting part, which is harmless for joint estimation.
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Segmentation:
    labels: np.ndarray  # (N,) part id, -1 = outlier
    poses: list[list[Pose]]  # [part][frame] mapping frame-0 coords -> frame t
    residual_rms: np.ndarray  # (n_parts,)

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
    """Per-point 90th-percentile residual over visible frames (robust to blips)."""
    masked = np.where(vis, res, np.nan)
    return np.nanpercentile(masked, 90, axis=0)


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
    full = ts.visible.all(axis=0)
    ref = ts.points[0]
    unassigned = np.ones(N, bool)
    parts: list[np.ndarray] = []

    while unassigned.sum() >= cfg.min_part_points and len(parts) < cfg.max_parts:
        cand = np.flatnonzero(unassigned & full)
        if len(cand) < cfg.sample_size:
            break
        seeds = rng.choice(cand, size=min(cfg.hypotheses, len(cand)), replace=len(cand) < cfg.hypotheses)
        d = np.linalg.norm(ref[cand][None, :, :] - ref[seeds][:, None, :], axis=2)
        k = cfg.sample_size
        nn = cand[np.argsort(d, axis=1)[:, :k]]  # (H,k) local neighbourhoods
        P = np.broadcast_to(ref[nn][:, None], (len(seeds), ts.n_frames, k, 3)).reshape(-1, k, 3)
        Q = np.transpose(ts.points[:, nn], (1, 0, 2, 3)).reshape(-1, k, 3)
        R, t = batch_kabsch(P, Q)
        R = R.reshape(len(seeds), ts.n_frames, 3, 3)
        t = t.reshape(len(seeds), ts.n_frames, 3)
        best_count, best_inl = -1, None
        for h in range(len(seeds)):
            score = _robust_point_score(_point_residuals(ts, R[h], t[h]), ts.visible)
            inl = (score < cfg.inlier_tol) & unassigned
            c = int(inl.sum())
            if c > best_count:
                best_count, best_inl = c, inl
        if best_count < cfg.min_part_points:
            break
        # refine: refit on the consensus set, re-collect inliers
        inl = best_inl
        for _ in range(2):
            poses = fit_part_poses(ts, np.flatnonzero(inl))
            Rr = np.stack([p.R for p in poses])
            tr = np.stack([p.t for p in poses])
            score = _robust_point_score(_point_residuals(ts, Rr, tr), ts.visible)
            inl = (score < cfg.inlier_tol) & unassigned
        if inl.sum() < cfg.min_part_points:
            break
        parts.append(np.flatnonzero(inl))
        unassigned &= ~inl

    if not parts:
        raise ValueError("segmentation found no rigid part; check units (metres) and noise level")

    # final assignment: each point to its best-fitting part
    part_poses = [fit_part_poses(ts, idx) for idx in parts]
    scores = np.stack([
        _robust_point_score(_point_residuals(ts, np.stack([p.R for p in pp]), np.stack([p.t for p in pp])), ts.visible)
        for pp in part_poses
    ])
    labels = np.argmin(scores, axis=0)
    labels[scores.min(axis=0) > 2 * cfg.inlier_tol] = -1
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
    labels = np.asarray(labels, int)
    if labels.shape != (ts.n_points,):
        raise ValueError("labels must have one entry per track")
    ids = sorted(int(i) for i in np.unique(labels) if i >= 0)
    remap = {old: new for new, old in enumerate(ids)}
    lab = np.array([remap.get(int(l), -1) for l in labels])
    poses = [fit_part_poses(ts, np.flatnonzero(lab == i)) for i in range(len(ids))]
    rms = np.array([_part_rms(ts, np.flatnonzero(lab == i), poses[i]) for i in range(len(ids))])
    return Segmentation(lab, poses, rms)
