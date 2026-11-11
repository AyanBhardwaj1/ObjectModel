"""Kinematic tree inference over segmented parts.

Root = the part that moves least in the world (assumes a static base, the
common case for furniture/appliances). Edges are added Prim-style: at each step
attach the out-of-tree part whose best joint to an in-tree part has the lowest
cost, where cost = joint residual + a spatial-gap penalty (joints connect
touching parts). A serial chain (lamp) therefore wins over a direct base→tip
"joint", because the composition of two hinges is not a single hinge.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .joints import JointEstimate, JointFitConfig, fit_joint
from .segmentation import Segmentation
from .synthetic import TrackSet

GAP_WEIGHT = 0.05  # metres of residual charged per metre of separation


@dataclass
class TreeEdge:
    parent: int
    child: int
    joint: JointEstimate
    cost: float


@dataclass
class KinematicTree:
    root: int
    edges: list[TreeEdge]

    def parent_of(self, child: int) -> int | None:
        for e in self.edges:
            if e.child == child:
                return e.parent
        return None

    def order(self) -> list[int]:
        """Parts in parent-before-child order."""
        out = [self.root]
        for e in self.edges:  # Prim order is already topological
            out.append(e.child)
        return out


def part_motion(ts: TrackSet, seg: Segmentation, i: int) -> float:
    pts = ts.points[0, seg.labels == i]
    return float(np.mean([np.linalg.norm(p.apply(pts) - pts, axis=1).mean() for p in seg.poses[i]]))


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    a = a[:: max(1, len(a) // 150)]
    b = b[:: max(1, len(b) // 150)]
    return float(np.min(np.linalg.norm(a[:, None] - b[None], axis=2)))


def infer_tree(ts: TrackSet, seg: Segmentation, cfg: JointFitConfig = JointFitConfig()) -> KinematicTree:
    n = seg.n_parts
    pts0 = [ts.points[0, seg.labels == i] for i in range(n)]
    motion = [part_motion(ts, seg, i) for i in range(n)]
    sizes = [len(p) for p in pts0]
    root = int(min(range(n), key=lambda i: (round(motion[i], 4), -sizes[i])))
    fast = replace(cfg, bootstrap=0)
    cache: dict[tuple[int, int], tuple[float, JointEstimate]] = {}

    def edge(i: int, j: int) -> tuple[float, JointEstimate]:
        if (i, j) not in cache:
            est = fit_joint(seg.poses[i], seg.poses[j], pts0[j], fast)
            cache[(i, j)] = (est.rms + GAP_WEIGHT * _gap(pts0[i], pts0[j]), est)
        return cache[(i, j)]

    in_tree, edges = {root}, []
    while len(in_tree) < n:
        best = min(
            ((edge(i, j)[0], i, j) for i in in_tree for j in range(n) if j not in in_tree),
            key=lambda x: x[0],
        )
        cost, i, j = best
        est = fit_joint(seg.poses[i], seg.poses[j], pts0[j], cfg)  # full fit with bootstrap
        edges.append(TreeEdge(i, j, est, cost))
        in_tree.add(j)
    return KinematicTree(root, edges)
