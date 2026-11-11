"""The articulated model: links, joints, and conversion from estimates.

Conventions: every link frame is world-axis-aligned at the rest pose (frame 0).
A child link's frame sits on its joint (pivot for hinges, part centroid for
sliders), so URDF joint origins are pure translations and axes are expressed
in world/rest coordinates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .joints import JointEstimate
from .physics import PartGeometry, PhysicalProps, part_geometry, physical_props
from .segmentation import Segmentation
from .synthetic import TrackSet
from .tree import KinematicTree

NAME_RE = re.compile(r"[^A-Za-z0-9_]+")
RESERVED_NAMES = {"world"}  # MuJoCo's implicit world body; URDF world convention


def safe_name(name: str, fallback: str) -> str:
    s = NAME_RE.sub("_", str(name or "")).strip("_")[:48]
    if not s or not s[0].isalpha():
        s = f"{fallback}_{s}" if s else fallback
    if s.lower() in RESERVED_NAMES:
        s = f"{fallback}_{s}"
    return s


@dataclass
class Link:
    name: str
    part_id: int
    origin: np.ndarray  # world position of the link frame at rest
    geometry: PartGeometry
    physics: PhysicalProps
    points: np.ndarray  # rest-pose sample points (world), for validation/viewer


@dataclass
class Joint:
    name: str
    type: str  # revolute | prismatic | continuous | fixed
    parent: str
    child: str
    axis: np.ndarray
    origin: np.ndarray  # world position at rest (== child link origin)
    lower: float
    upper: float
    states: np.ndarray
    confidence: float
    axis_std_deg: float
    origin_std: float
    rms: float
    damping: float = 0.05
    friction: float = 0.0
    type_probs: dict = field(default_factory=dict)


@dataclass
class ArticulatedModel:
    name: str
    links: list[Link]
    joints: list[Joint]
    notes: list[str] = field(default_factory=list)

    def link(self, name: str) -> Link:
        return next(l for l in self.links if l.name == name)

    def children(self, name: str) -> list[Joint]:
        return [j for j in self.joints if j.parent == name]

    @property
    def root(self) -> Link:
        childs = {j.child for j in self.joints}
        return next(l for l in self.links if l.name not in childs)


def _outward(est: JointEstimate, away: np.ndarray) -> JointEstimate:
    """Orient slider axes away from the parent so positive values mean 'pull out'."""
    from dataclasses import replace

    if est.type != "prismatic" or float(np.dot(est.axis, away)) >= 0:
        return est
    return replace(est, axis=-est.axis, states=-est.states, lower=-est.upper, upper=-est.lower)


def _joint_type(est: JointEstimate) -> str:
    if est.type == "revolute" and (est.upper - est.lower) > np.radians(350):
        return "continuous"
    return est.type


def build_model(
    ts: TrackSet,
    seg: Segmentation,
    tree: KinematicTree,
    name: str = "object",
    part_names: dict[int, str] | None = None,
    materials: dict[int, tuple[str, float | None]] | None = None,
    joint_names: dict[int, str] | None = None,
) -> ArticulatedModel:
    part_names = part_names or {}
    materials = materials or {}
    joint_names = joint_names or {}
    used: set[str] = set()

    def uniq(n: str) -> str:
        base, k = n, 2
        while n in used:
            n = f"{base}_{k}"
            k += 1
        used.add(n)
        return n

    names = {i: uniq(safe_name(part_names.get(i, f"part{i}"), f"part{i}")) for i in range(seg.n_parts)}
    edge_of = {e.child: e for e in tree.edges}
    links = []
    for i in tree.order():
        pts = ts.points[0, seg.labels == i]
        geom = part_geometry(pts)
        mat, fill = materials.get(i, ("plastic", None))
        e = edge_of.get(i)
        # Root frame = capture world origin, so URDF and MJCF place every link identically.
        origin = np.zeros(3) if e is None else np.asarray(e.joint.origin, float)
        links.append(Link(names[i], i, origin, geom, physical_props(geom, mat, fill), pts))
    centroids = {i: ts.points[0, seg.labels == i].mean(axis=0) for i in range(seg.n_parts)}
    joints = []
    for e in tree.edges:
        est = _outward(e.joint, centroids[e.child] - centroids[e.parent])
        joints.append(Joint(
            name=uniq(safe_name(joint_names.get(e.child, f"{names[e.child]}_joint"), "joint")),
            type=_joint_type(est), parent=names[e.parent], child=names[e.child],
            axis=np.asarray(est.axis, float), origin=np.asarray(est.origin, float),
            lower=est.lower, upper=est.upper, states=est.states, confidence=est.confidence,
            axis_std_deg=est.axis_std_deg, origin_std=est.origin_std, rms=est.rms,
            type_probs=est.type_probs,
        ))
    return ArticulatedModel(safe_name(name, "object"), links, joints)
