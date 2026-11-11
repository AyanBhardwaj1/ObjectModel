"""Run humanoid manipulation tasks on articulated objects: plan from the twin, score in the truth.

    from objectmodel.tasks import run_task
    results = run_task("kitchen_cabinet", perturb_deg=0)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import synthetic as s
from .export import to_mjcf
from .geometry import axis_angle_to_matrix
from .humanoid import TaskResult, closed_and_open, execute
from .model import ArticulatedModel, Joint, Link
from .physics import PartGeometry, physical_props
from .pipeline import run


def ground_truth_model(obj: s.SyntheticObject, ts: s.TrackSet) -> ArticulatedModel:
    """The object as it really is, in the same rest frame (capture frame 0) as the twin."""
    links = []
    by_child = {j.child: j for j in obj.joints}
    for i, p in enumerate(obj.parts):
        geom = PartGeometry(np.asarray(p.center, float), np.eye(3), np.asarray(p.size, float))
        j = by_child.get(p.name)
        origin = np.zeros(3) if j is None else np.asarray(j.origin, float)
        links.append(Link(p.name, i, origin, geom, physical_props(geom, p.material), np.zeros((0, 3))))
    joints = []
    for j in obj.joints:
        q = ts.joint_states[j.child]
        joints.append(Joint(f"{j.child}_joint", j.type, j.parent, j.child, np.asarray(j.axis, float) / np.linalg.norm(j.axis),
                            np.asarray(j.origin, float), float(q.min()), float(q.max()), q, 1.0, 0.0, 0.0, 0.0))
    return ArticulatedModel(f"{obj.name}_truth", links, joints)


def perturb(j: Joint, degrees: float) -> Joint:
    """Rotate a joint axis by `degrees` about a perpendicular direction (a deliberately wrong twin)."""
    if not degrees:
        return j
    a = j.axis / np.linalg.norm(j.axis)
    perp = np.cross(a, [0, 0, 1]) if abs(a[2]) < 0.9 else np.cross(a, [1, 0, 0])
    perp /= np.linalg.norm(perp)
    return replace(j, axis=axis_angle_to_matrix(perp, np.radians(degrees)) @ a)


@dataclass
class TaskCase:
    object: str
    joint: str
    result: TaskResult  # planned with the measured twin (optionally perturbed)
    oracle: TaskResult | None  # same task planned with the true joint
    axis_error_deg: float

    @property
    def excess_peak_n(self) -> float:
        """Grasp force above the true-joint plan (signed: run-to-run noise can make it slightly negative)."""
        base = self.oracle.peak_force_n if self.oracle else 0.0
        return self.result.peak_force_n - base

    @property
    def feasible(self) -> bool:
        """Can this robot and controller do the task at all? (The plan with the true joint succeeds.)"""
        return self.oracle is None or self.oracle.success

    @property
    def success(self) -> bool:
        from .humanoid import judge
        return self.feasible and judge(self.result, self.oracle)


def _owner(labels: np.ndarray, gt: np.ndarray, part: int) -> int:
    vals, counts = np.unique(gt[(labels == part) & (gt >= 0)], return_counts=True)
    return int(vals[np.argmax(counts)])


def _lateral_shift(model: ArticulatedModel, joint: Joint, metres: float) -> np.ndarray:
    """A sideways offset along the part's face (perpendicular to the approach), for localisation studies."""
    from .humanoid import affordance
    n = affordance(model, joint).normal
    side = np.cross([0, 0, 1], n)
    return side / max(np.linalg.norm(side), 1e-9) * metres


def work_height_lift(truth: ArticulatedModel, joint: Joint, robot) -> float:
    """Standing humanoids here do not squat: raise low handles to about hip height on a stand."""
    if robot.adjustable_height:
        return 0.0
    from .humanoid import affordance
    z = affordance(truth, joint).handle[2]
    return float(max(0.0, 0.95 - z))


def run_task(object_name: str, seed: int = 0, noise_mm: float = 1.0, perturb_deg: float = 0.0,
             joints: list[str] | None = None, render: bool = False, oracle: bool = True,
             robot: str = "generic", lift: float | None = None, fps: int = 30, size=(960, 540),
             log: bool = False, handle_shift: float = 0.0) -> list[TaskCase]:
    """Open every first-level joint of an object (or the named GT children) with the humanoid."""
    from .geometry import angle_between_axes
    from .robots import get_robot

    bot = get_robot(robot)

    obj = s.CATALOGUE[object_name]()
    ts = s.generate_tracks(obj, seed=seed, noise=noise_mm / 1000)
    twin_res = run(ts, hint=object_name.replace("_", " "), use_llm=False)
    twin = twin_res.model
    truth = ground_truth_model(obj, ts)
    world_xml = to_mjcf(truth)
    root = truth.root.name
    cases = []
    for tj in twin.joints:
        if tj.parent != twin.root.name or tj.type not in ("revolute", "prismatic"):
            continue
        gt_part = obj.parts[_owner(twin_res.segmentation.labels, ts.labels, twin.link(tj.child).part_id)].name
        gj = next((g for g in truth.joints if g.child == gt_part and g.parent == root), None)
        if gj is None or (joints and gt_part not in joints):
            continue
        plan_joint = perturb(tj, perturb_deg)
        plan = replace(twin, joints=[plan_joint if j.name == tj.name else j for j in twin.joints])
        wq = closed_and_open(truth, gj)
        h = lift if lift is not None else work_height_lift(twin, tj, bot)  # chosen from the twin only
        res = execute(world_xml, gj.child, gj.name, wq, plan, plan_joint, robot=bot, lift=h, render=render, fps=fps, size=size, log=log,
                      handle_shift=None if not handle_shift else _lateral_shift(plan, plan_joint, handle_shift))
        ora = execute(world_xml, gj.child, gj.name, wq, truth, gj, robot=bot, lift=h) if oracle else None
        cases.append(TaskCase(object_name, gt_part, res, ora, angle_between_axes(plan_joint.axis, gj.axis)))
    return cases
