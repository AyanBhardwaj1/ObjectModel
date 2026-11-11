"""How accurate must a twin be for a humanoid to use it?

Sweep a deliberate hinge/slide-axis error over every openable joint, run the humanoid task planned
from that twin in the ground-truth world, and record the grasp force the error adds on top of what the
task itself needs (the oracle plan with the true joint).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .tasks import run_task

DEFAULT_OBJECTS = ("kitchen_cabinet", "fridge", "dresser", "drawer_cabinet", "laptop")
DEFAULT_ERRORS = (0.0, 2.0, 5.0, 10.0, 15.0)


@dataclass
class Row:
    object: str
    joint: str
    kind: str
    added_error_deg: float
    axis_error_deg: float  # measured twin error plus the added error
    opened: float
    peak_n: float
    oracle_peak_n: float
    excess_n: float
    track_mm: float
    excess_track_mm: float  # hand-path deviation above the true-joint plan (compliant arms show error here)
    feasible: bool  # the same task planned with the true joint succeeds on this robot
    grasped: bool
    success: bool


def sweep(objects=DEFAULT_OBJECTS, errors=DEFAULT_ERRORS, robot: str = "generic", seed: int = 0) -> list[Row]:
    rows: list[Row] = []
    for obj in objects:
        oracles: dict[str, object] = {}
        for err in errors:
            for c in run_task(obj, seed=seed, perturb_deg=err, robot=robot, oracle=not oracles):
                if c.oracle is not None:
                    oracles[c.joint] = c.oracle
                o = oracles.get(c.joint)
                base = o.peak_force_n if o is not None else 0.0
                feasible = bool(o is None or o.success)
                r = c.result
                from .humanoid import judge
                rows.append(Row(obj, c.joint, r.kind, err, round(c.axis_error_deg, 2), round(r.opened_fraction, 3),
                                round(r.peak_force_n, 1), round(base, 1), round(r.peak_force_n - base, 1),
                                round(r.max_track_mm, 1), round(r.max_track_mm - (o.max_track_mm if o else 0.0), 1),
                                feasible, r.grasped, bool(feasible and judge(r, o))))
    return rows


def summary_table(rows: list[Row]) -> str:
    """Only tasks the robot can do with the true joint count; the rest are listed as infeasible."""
    errs = sorted({r.added_error_deg for r in rows})
    infeasible = sorted({f"{r.object}/{r.joint}" for r in rows if not r.feasible})
    lines = ["| axis error | feasible tasks | succeed | grasp missed | median extra grasp force | worst extra grasp force | median extra hand deviation |",
             "|---|---|---|---|---|---|---|"]
    for e in errs:
        rs = [r for r in rows if r.added_error_deg == e and r.feasible]
        ex = [r.excess_n for r in rs if r.grasped] or [float("nan")]
        label = "measured twin" if e == 0 else f"twin + {e:g}°"
        missed = sum(not r.grasped for r in rs)
        dev = [r.excess_track_mm for r in rs if r.grasped] or [float("nan")]
        lines.append(f"| {label} | {len(rs)} | {np.mean([r.success for r in rs]):.0%} | {missed} | "
                     f"{np.nanmedian(ex):+.0f} N | {np.nanmax(ex):+.0f} N | {np.nanmedian(dev):+.0f} mm |")
    if infeasible:
        lines.append(f"\nInfeasible for this robot even with the true joint (excluded): {', '.join(infeasible)}.")
    return "\n".join(lines)


def by_kind_table(rows: list[Row]) -> str:
    errs = sorted({r.added_error_deg for r in rows})
    lines = ["| axis error | drawers and slides: extra force | doors and lids: extra force |", "|---|---|---|"]
    for e in errs:
        p = [r.excess_n for r in rows if r.added_error_deg == e and r.kind == "pull" and r.feasible and r.grasped]
        s = [r.excess_n for r in rows if r.added_error_deg == e and r.kind == "swing" and r.feasible and r.grasped]
        f = lambda xs: f"{np.median(xs):+.0f} N (max {np.max(xs):+.0f})" if xs else "n/a"
        label = "measured twin" if e == 0 else f"twin + {e:g}°"
        lines.append(f"| {label} | {f(p)} | {f(s)} |")
    return "\n".join(lines)


def to_json(rows: list[Row]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=1)
