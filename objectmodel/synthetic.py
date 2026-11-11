"""Procedural articulated objects with exact ground truth.

Used for tests, the benchmark, and demos. Each object is a kinematic tree of
box-shaped parts; we sample surface points, animate the joints, and emit noisy
3D point tracks — the same input a real tracker + depth pipeline produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose, axis_angle_to_matrix


@dataclass(frozen=True)
class BoxPart:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str = "plastic"


@dataclass(frozen=True)
class GTJoint:
    parent: str
    child: str
    type: str  # revolute | prismatic
    axis: tuple[float, float, float]
    origin: tuple[float, float, float]
    lower: float
    upper: float


@dataclass
class SyntheticObject:
    name: str
    parts: list[BoxPart]
    joints: list[GTJoint]
    description: str = ""

    def part_index(self, name: str) -> int:
        return [p.name for p in self.parts].index(name)


@dataclass
class TrackSet:
    """Point tracks: points (T,N,3) in metres, visible (T,N) bool."""

    points: np.ndarray
    visible: np.ndarray
    labels: np.ndarray | None = None  # ground-truth part ids (N,) when known
    joint_states: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError(f"points must be (T,N,3), got {self.points.shape}")
        if self.visible.shape != self.points.shape[:2]:
            raise ValueError("visible must be (T,N) matching points")
        if not np.isfinite(self.points[self.visible]).all():
            raise ValueError("visible points must be finite")

    @property
    def n_frames(self) -> int:
        return self.points.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[1]

    def save(self, path: str) -> None:
        extra = {} if self.labels is None else {"labels": self.labels}
        np.savez_compressed(path, points=self.points, visible=self.visible, **extra)

    @staticmethod
    def load(path: str) -> "TrackSet":
        with np.load(path, allow_pickle=False) as z:
            pts = np.asarray(z["points"], float)
            vis = np.asarray(z["visible"], bool) if "visible" in z else np.ones(pts.shape[:2], bool)
            labels = np.asarray(z["labels"]) if "labels" in z else None
        pts = np.where(vis[..., None], pts, 0.0)
        return TrackSet(pts, vis, labels)


def sample_box_surface(center, size, n: int, rng: np.random.Generator) -> np.ndarray:
    size = np.asarray(size, float)
    areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
    face_axis = rng.choice(3, size=n, p=np.repeat(areas, 1) / areas.sum())
    pts = (rng.random((n, 3)) - 0.5) * size
    sign = rng.choice([-0.5, 0.5], size=n)
    pts[np.arange(n), face_axis] = sign * size[face_axis]
    return pts + np.asarray(center, float)


def joint_motion(j: GTJoint, q: float) -> Pose:
    axis = np.asarray(j.axis, float) / np.linalg.norm(j.axis)
    o = np.asarray(j.origin, float)
    if j.type == "revolute":
        R = axis_angle_to_matrix(axis, q)
        return Pose(R, o - R @ o)
    if j.type == "prismatic":
        return Pose(np.eye(3), q * axis)
    raise ValueError(f"unknown joint type {j.type}")


def _smooth_profile(T: int, lower: float, upper: float, phase: float, rng) -> np.ndarray:
    """Start at the rest pose (0, the modelled geometry) and swing toward the far limit and back.

    Starting at 0 keeps frame 0 identical to the part geometry, so a door whose range is
    negative opens outward instead of being re-zeroed into the cabinet.
    """
    s = np.linspace(0, 1, T)
    far = lower if abs(lower) > abs(upper) else upper
    if not (lower <= 0.0 <= upper):
        far = upper
    delay = phase % 0.3  # joints start at different times so their motions are distinct
    u = np.clip((s - delay) / (1.0 - delay), 0.0, 1.0)
    wave = 0.5 - 0.5 * np.cos(2 * np.pi * u)
    return 0.9 * far * wave


def animate(obj: SyntheticObject, T: int, rng: np.random.Generator) -> tuple[list[list[Pose]], dict]:
    names = [p.name for p in obj.parts]
    by_child = {j.child: j for j in obj.joints}
    states = {}
    for k, j in enumerate(obj.joints):
        q = _smooth_profile(T, j.lower, j.upper, phase=0.17 * k, rng=rng)  # q[0] == 0: rest pose = frame 0
        states[j.child] = q
    poses: list[list[Pose]] = [[Pose.identity()] * len(names) for _ in range(T)]
    order = _topological(obj)
    for t in range(T):
        frame = {}
        for name in order:
            j = by_child.get(name)
            if j is None:
                frame[name] = Pose.identity()
            else:
                frame[name] = frame[j.parent].compose(joint_motion(j, states[name][t]))
        poses[t] = [frame[n] for n in names]
    return poses, states


def _topological(obj: SyntheticObject) -> list[str]:
    children: dict[str, list[str]] = {}
    child_set = set()
    for j in obj.joints:
        children.setdefault(j.parent, []).append(j.child)
        child_set.add(j.child)
    roots = [p.name for p in obj.parts if p.name not in child_set]
    out, stack = [], list(roots)
    while stack:
        n = stack.pop(0)
        out.append(n)
        stack.extend(children.get(n, []))
    return out


def generate_tracks(
    obj: SyntheticObject,
    n_frames: int = 48,
    points_per_part: int = 160,
    noise: float = 0.001,
    dropout: float = 0.05,
    outlier_frac: float = 0.0,
    seed: int = 0,
    camera_motion: float = 0.0,
) -> TrackSet:
    """camera_motion > 0 simulates a hand-held camera: the whole scene undergoes a
    smooth rigid motion (rotation up to camera_motion rad, translation up to
    camera_motion * 0.5 m), so no part is static in the observed frame."""
    rng = np.random.default_rng(seed)
    clouds, labels = [], []
    for i, p in enumerate(obj.parts):
        clouds.append(sample_box_surface(p.center, p.size, points_per_part, rng))
        labels.append(np.full(points_per_part, i))
    base = np.concatenate(clouds)
    labels = np.concatenate(labels)
    poses, states = animate(obj, n_frames, rng)
    pts = np.empty((n_frames, len(base), 3))
    for t in range(n_frames):
        for i in range(len(obj.parts)):
            m = labels == i
            pts[t, m] = poses[t][i].apply(base[m])
    if camera_motion > 0:
        s_ = np.linspace(0, 1, n_frames)
        ax = rng.normal(size=3)
        drift = rng.normal(size=3)
        drift /= np.linalg.norm(drift)
        c = base.mean(axis=0)
        for t in range(n_frames):
            R = axis_angle_to_matrix(ax, camera_motion * np.sin(np.pi * s_[t]))
            pts[t] = (pts[t] - c) @ R.T + c + drift * camera_motion * 0.5 * s_[t]
    pts += rng.normal(0, noise, pts.shape)
    if outlier_frac > 0:
        k = int(outlier_frac * len(base))
        idx = rng.choice(len(base), k, replace=False)
        pts[:, idx] += rng.normal(0, 0.05, (n_frames, k, 3)).cumsum(axis=0) * 0.2
        labels = labels.copy()
        labels[idx] = -1
    visible = rng.random(pts.shape[:2]) > dropout
    visible[0] = True
    pts = np.where(visible[..., None], pts, 0.0)
    return TrackSet(pts, visible, labels, states, {"object": obj.name})


# ---------------------------------------------------------------- catalogue

def laptop() -> SyntheticObject:
    return SyntheticObject(
        "laptop",
        [BoxPart("base", (0, 0, 0.0075), (0.304, 0.212, 0.015), "aluminum"),
         BoxPart("lid", (0, 0.104, 0.115), (0.304, 0.006, 0.2), "aluminum")],
        [GTJoint("base", "lid", "revolute", (1, 0, 0), (0, 0.106, 0.015), -1.2, 0.6)],
        "13-inch laptop, lid on a rear hinge",
    )


def drawer_cabinet() -> SyntheticObject:
    return SyntheticObject(
        "drawer_cabinet",
        [BoxPart("body", (0, 0, 0.3), (0.45, 0.4, 0.6), "wood"),
         BoxPart("drawer", (0, -0.21, 0.45), (0.38, 0.03, 0.16), "wood")],
        [GTJoint("body", "drawer", "prismatic", (0, -1, 0), (0, -0.21, 0.45), 0.0, 0.3)],
        "bedside cabinet with one pull-out drawer",
    )


def desk_lamp() -> SyntheticObject:
    return SyntheticObject(
        "desk_lamp",
        [BoxPart("base", (0, 0, 0.015), (0.16, 0.16, 0.03), "steel"),
         BoxPart("lower_arm", (0, 0, 0.2), (0.025, 0.025, 0.34), "aluminum"),
         BoxPart("upper_arm", (0, 0.15, 0.37), (0.025, 0.3, 0.025), "aluminum")],
        [GTJoint("base", "lower_arm", "revolute", (1, 0, 0), (0, 0, 0.03), -0.6, 0.5),
         GTJoint("lower_arm", "upper_arm", "revolute", (1, 0, 0), (0, 0, 0.37), -0.8, 0.7)],
        "two-link desk lamp (serial chain)",
    )


def kitchen_cabinet() -> SyntheticObject:
    return SyntheticObject(
        "kitchen_cabinet",
        [BoxPart("body", (0, 0, 0.4), (0.6, 0.5, 0.8), "wood"),
         BoxPart("door", (-0.15, -0.26, 0.55), (0.3, 0.02, 0.45), "wood"),
         BoxPart("drawer", (0.15, -0.26, 0.6), (0.28, 0.02, 0.14), "wood")],
        [GTJoint("body", "door", "revolute", (0, 0, 1), (-0.3, -0.26, 0.55), -1.6, 0.0),
         GTJoint("body", "drawer", "prismatic", (0, -1, 0), (0.15, -0.26, 0.6), 0.0, 0.25)],
        "cabinet with a hinged door and a drawer (branching tree)",
    )


def faucet() -> SyntheticObject:
    return SyntheticObject(
        "faucet",
        [BoxPart("body", (0, 0, 0.1), (0.05, 0.05, 0.2), "steel"),
         BoxPart("spout", (0, 0.1, 0.19), (0.03, 0.18, 0.03), "steel"),
         BoxPart("handle", (0, -0.03, 0.23), (0.02, 0.08, 0.02), "steel")],
        [GTJoint("body", "spout", "revolute", (0, 0, 1), (0, 0, 0.19), -0.8, 0.8),
         GTJoint("body", "handle", "revolute", (1, 0, 0), (0, 0, 0.22), -0.5, 0.0)],
        "kitchen faucet with swivel spout and lever",
    )


def dresser() -> SyntheticObject:
    drawers = [BoxPart(f"drawer{k}", (0.15, -0.27, 0.2 + 0.22 * k), (0.32, 0.03, 0.18), "wood") for k in range(3)]
    return SyntheticObject(
        "dresser",
        [BoxPart("body", (0, 0, 0.4), (0.8, 0.5, 0.8), "wood"), *drawers,
         BoxPart("door", (-0.2, -0.265, 0.4), (0.36, 0.02, 0.7), "wood")],
        [*(GTJoint("body", f"drawer{k}", "prismatic", (0, -1, 0), (0.15, -0.27, 0.2 + 0.22 * k), 0.0, 0.3)
           for k in range(3)),
         GTJoint("body", "door", "revolute", (0, 0, 1), (-0.38, -0.265, 0.4), -1.7, 0.0)],
        "dresser: three drawers and a hinged door (5 parts, 4 joints)",
    )


def fridge() -> SyntheticObject:
    return SyntheticObject(
        "fridge",
        [BoxPart("body", (0, 0, 0.9), (0.7, 0.65, 1.8), "steel"),
         BoxPart("left_door", (-0.175, -0.34, 0.9), (0.35, 0.04, 1.7), "steel"),
         BoxPart("right_door", (0.175, -0.34, 0.9), (0.35, 0.04, 1.7), "steel")],
        [GTJoint("body", "left_door", "revolute", (0, 0, 1), (-0.35, -0.34, 0.9), -1.9, 0.0),
         GTJoint("body", "right_door", "revolute", (0, 0, 1), (0.35, -0.34, 0.9), 0.0, 1.9)],
        "French-door fridge with mirrored hinges",
    )


CATALOGUE = {
    "laptop": laptop,
    "drawer_cabinet": drawer_cabinet,
    "desk_lamp": desk_lamp,
    "kitchen_cabinet": kitchen_cabinet,
    "faucet": faucet,
    "dresser": dresser,
    "fridge": fridge,
}
