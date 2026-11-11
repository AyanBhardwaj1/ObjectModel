"""Simulator-in-the-loop validation of an exported model with MuJoCo.

1. Compile check: the MJCF must load (catches bad inertia, limits, names).
2. Static checks: positive masses, valid inertia, observed states inside limits.
3. Kinematic replay: drive the compiled model with the estimated joint states and
   compare every simulated part surface point against the observed tracks.
4. Passive stability: simulate under gravity and require finite, bounded motion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .export import to_mjcf
from .model import ArticulatedModel
from .synthetic import TrackSet

REPLAY_PASS_MM = 10.0
STABILITY_SECONDS = 2.0
MAX_STABLE_SPEED = 50.0  # rad/s or m/s


@dataclass
class ValidationReport:
    compiled: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    replay_rms_mm: dict = field(default_factory=dict)
    replay_overall_mm: float | None = None
    noise_mm: float = 0.0  # estimated per-axis track noise
    replay_excess_mm: float | None = None  # replay error not explained by measurement noise
    stable: bool | None = None
    simulator: str = "mujoco"

    @property
    def passed(self) -> bool:
        return (
            self.compiled
            and not self.errors
            and self.replay_excess_mm is not None
            and self.replay_excess_mm < REPLAY_PASS_MM
            and bool(self.stable)
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def static_checks(model: ArticulatedModel) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for l in model.links:
        I = l.physics.inertia_diag
        if l.physics.mass <= 0:
            errors.append(f"link {l.name}: non-positive mass")
        if np.any(I <= 0) or I[0] + I[1] < I[2] or I[0] + I[2] < I[1] or I[1] + I[2] < I[0]:
            errors.append(f"link {l.name}: inertia violates triangle inequality")
    for j in model.joints:
        if j.type in ("revolute", "prismatic") and not (j.lower < j.upper):
            errors.append(f"joint {j.name}: empty limit range")
        if j.confidence < 0.9:
            warnings.append(f"joint {j.name}: low type confidence ({j.confidence:.2f})")
        if j.type != "fixed" and not np.isfinite(j.axis_std_deg):
            warnings.append(f"joint {j.name}: uncertainty not estimated (too few frames)")
        elif j.axis_std_deg > 5:
            warnings.append(f"joint {j.name}: uncertain axis (±{j.axis_std_deg:.1f}°) — record a wider motion range")
    return errors, warnings


def validate(model: ArticulatedModel, ts: TrackSet, labels: np.ndarray, noise_sigma: float | None = None) -> ValidationReport:
    from .segmentation import estimate_noise

    sigma = estimate_noise(ts) if noise_sigma is None else noise_sigma
    errors, warnings = static_checks(model)
    try:
        import mujoco
    except ImportError:
        return ValidationReport(False, errors, warnings + ["mujoco not installed; simulator checks skipped"])
    try:
        m = mujoco.MjModel.from_xml_string(to_mjcf(model))
    except Exception as exc:  # compile errors are the finding we report
        return ValidationReport(False, errors + [f"MJCF failed to compile: {exc}"], warnings)
    d = mujoco.MjData(m)
    report = ValidationReport(True, errors, warnings, noise_mm=round(sigma * 1000, 3))
    _replay(model, ts, labels, m, d, report, mujoco)
    if report.replay_overall_mm is not None:
        # Rest-frame and frame-t observations are both noisy: a perfect model still
        # shows sqrt(2 frames * 3 axes) * sigma of replay error.
        expected = np.sqrt(6.0) * sigma * 1000
        report.replay_excess_mm = round(float(np.sqrt(max(0.0, report.replay_overall_mm ** 2 - expected ** 2))), 3)
    report.stable = _stability(model, m, mujoco)
    if not report.stable:
        report.errors.append("passive simulation diverged")
    return report


def _set_qpos(model, m, d, t: int, mujoco) -> None:
    d.qpos[:] = 0
    for j in model.joints:
        if j.type == "fixed":
            continue
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)
        d.qpos[m.jnt_qposadr[jid]] = j.states[t]


def _in_root_frame(model, ts: TrackSet, labels: np.ndarray) -> np.ndarray:
    """Observed points expressed in the root part's frame-0 coordinates.

    The simulated root is welded to the world; a hand-held camera moves the whole
    scene, so we remove the root's tracked motion before comparing. For a static
    camera the root pose is ~identity and this is a no-op.
    """
    from .segmentation import fit_part_poses

    root_idx = np.flatnonzero(labels == model.root.part_id)
    poses = fit_part_poses(ts, root_idx)
    return np.stack([poses[t].inverse().apply(ts.points[t]) for t in range(ts.n_frames)])


def _replay(model, ts, labels, m, d, report, mujoco) -> None:
    obs = _in_root_frame(model, ts, labels)
    mujoco.mj_kinematics(m, d)
    rest = {}
    for l in model.links:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l.name)
        pts = obs[0, labels == l.part_id]
        rest[l.name] = (bid, (pts - d.xpos[bid]) @ d.xmat[bid].reshape(3, 3), np.flatnonzero(labels == l.part_id))
    sq = {l.name: [] for l in model.links}
    for t in range(ts.n_frames):
        _set_qpos(model, m, d, t, mujoco)
        mujoco.mj_kinematics(m, d)
        for name, (bid, local, idx) in rest.items():
            pred = local @ d.xmat[bid].reshape(3, 3).T + d.xpos[bid]
            vis = ts.visible[t, idx]
            if vis.any():
                sq[name].append(np.sum((pred[vis] - obs[t, idx][vis]) ** 2, axis=1))
    allsq = []
    for name, chunks in sq.items():
        if chunks:
            c = np.concatenate(chunks)
            allsq.append(c)
            report.replay_rms_mm[name] = round(float(np.sqrt(c.mean())) * 1000, 3)
    report.replay_overall_mm = round(float(np.sqrt(np.concatenate(allsq).mean())) * 1000, 3) if allsq else None
    for j in model.joints:
        if j.type in ("revolute", "prismatic"):
            out = np.mean((j.states < j.lower - 1e-6) | (j.states > j.upper + 1e-6))
            if out > 0:
                report.warnings.append(f"joint {j.name}: {out:.0%} of observed states outside limits")


def _stability(model, m, mujoco) -> bool:
    d = mujoco.MjData(m)
    for j in model.joints:  # start mid-range so limits and gravity both act
        if j.type in ("revolute", "prismatic"):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            d.qpos[m.jnt_qposadr[jid]] = 0.5 * (j.lower + j.upper)
    steps = int(STABILITY_SECONDS / m.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return bool(np.all(np.isfinite(d.qpos)) and np.all(np.abs(d.qvel) < MAX_STABLE_SPEED))
