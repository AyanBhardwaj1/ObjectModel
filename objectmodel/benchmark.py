"""Synthetic benchmark with Ditto/PARIS-style articulation metrics.

Per joint: type accuracy, axis angle error (deg), pivot distance (mm, revolute),
joint-state error; per object: topology accuracy, sim replay error, pass rate.
Run across noise levels to measure robustness.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import synthetic as s
from .geometry import angle_between_axes, point_line_distance
from .pipeline import run


@dataclass
class CaseResult:
    object: str
    noise_mm: float
    seed: int
    topology_ok: bool
    type_acc: float
    axis_err_deg: float
    pivot_err_mm: float
    state_err: float
    replay_mm: float
    sim_pass: bool
    seconds: float


def _owner(gt_labels, mask) -> int:
    vals, counts = np.unique(gt_labels[mask & (gt_labels >= 0)], return_counts=True)
    return int(vals[np.argmax(counts)]) if len(vals) else -1


def evaluate(name: str, noise: float, seed: int, camera_motion: float = 0.0) -> CaseResult:
    obj = s.CATALOGUE[name]()
    ts = s.generate_tracks(obj, noise=noise, seed=seed, camera_motion=camera_motion)
    t0 = time.time()
    nan = float("nan")
    try:
        res = run(ts, hint=name.replace("_", " "), use_llm=False)
    except ValueError:
        # a reconstruction that refuses to produce a model scores as a full failure
        return CaseResult(name, noise * 1000, seed, False, 0.0, nan, nan, nan, nan, False, time.time() - t0)
    dt = time.time() - t0
    seg = res.segmentation
    owner = {i: _owner(ts.labels, seg.labels == i) for i in range(seg.n_parts)}
    gt = {(j.parent, j.child): j for j in obj.joints}
    got = {}
    for e in res.tree.edges:
        if owner[e.parent] >= 0 and owner[e.child] >= 0:
            got[(obj.parts[owner[e.parent]].name, obj.parts[owner[e.child]].name)] = e.joint
    topo = set(got) == set(gt)
    types, axes, pivots, states = [], [], [], []
    for key, j in gt.items():
        est = got.get(key)
        if est is None:
            types.append(0.0)
            continue
        types.append(float(est.type == j.type))
        axes.append(angle_between_axes(est.axis, j.axis))
        if j.type == "revolute":
            pivots.append(point_line_distance(est.origin, j.origin, j.axis) * 1000)
        states.append(float(np.mean(np.abs(est.states - ts.joint_states[j.child]))))
    return CaseResult(
        name, noise * 1000, seed, topo, float(np.mean(types)),
        float(np.mean(axes)) if axes else nan, float(np.mean(pivots)) if pivots else nan,
        float(np.mean(states)) if states else nan,
        res.validation.replay_excess_mm if res.validation.replay_excess_mm is not None else nan, res.validation.passed, dt,
    )


def run_benchmark(noises=(0.001, 0.003, 0.006), seeds=(0, 1, 2), objects=None, camera_motion: float = 0.0) -> list[CaseResult]:
    objects = objects or list(s.CATALOGUE)
    return [evaluate(o, n, sd, camera_motion) for n in noises for o in objects for sd in seeds]


def _nanmean(xs) -> float:
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def summarize_table(results: list[CaseResult]) -> str:
    lines = [
        "| noise | topology | joint type | axis err ° | pivot err mm | state err | unexplained replay mm | sim pass | s/object |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for noise in sorted({r.noise_mm for r in results}):
        rs = [r for r in results if r.noise_mm == noise]
        m = lambda k: _nanmean([getattr(r, k) for r in rs])
        lines.append(
            f"| {noise:.0f} mm | {np.mean([r.topology_ok for r in rs]):.0%} | {m('type_acc'):.0%} | "
            f"{m('axis_err_deg'):.2f} | {m('pivot_err_mm'):.1f} | {m('state_err'):.4f} | {m('replay_mm'):.2f} | "
            f"{np.mean([r.sim_pass for r in rs]):.0%} | {m('seconds'):.1f} |"
        )
    return "\n".join(lines)


def per_object_table(results: list[CaseResult]) -> str:
    lines = ["| object | topology | joint type | axis err ° | pivot err mm | sim pass |", "|---|---|---|---|---|---|"]
    for o in dict.fromkeys(r.object for r in results):
        rs = [r for r in results if r.object == o]
        lines.append(
            f"| {o} | {np.mean([r.topology_ok for r in rs]):.0%} | {np.mean([r.type_acc for r in rs]):.0%} | "
            f"{_nanmean([r.axis_err_deg for r in rs]):.2f} | {_nanmean([r.pivot_err_mm for r in rs]):.1f} | "
            f"{np.mean([r.sim_pass for r in rs]):.0%} |"
        )
    return "\n".join(lines)
