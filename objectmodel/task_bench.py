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
    success: bool


def sweep(objects=DEFAULT_OBJECTS, errors=DEFAULT_ERRORS, robot: str = "generic", seed: int = 0) -> list[Row]:
    rows: list[Row] = []
    for obj in objects:
        oracle_peaks: dict[str, float] = {}
        for err in errors:
            for c in run_task(obj, seed=seed, perturb_deg=err, robot=robot, oracle=not oracle_peaks):
                if c.oracle is not None:
                    oracle_peaks[c.joint] = c.oracle.peak_force_n
                base = oracle_peaks.get(c.joint, 0.0)
                r = c.result
                rows.append(Row(obj, c.joint, r.kind, err, round(c.axis_error_deg, 2), round(r.opened_fraction, 3),
                                round(r.peak_force_n, 1), round(base, 1), round(max(0.0, r.peak_force_n - base), 1),
                                round(r.max_track_mm, 1),
                                bool(r.opened_fraction >= 0.9 and r.peak_force_n <= max(40.0, 1.25 * base))))
    return rows


def summary_table(rows: list[Row]) -> str:
    errs = sorted({r.added_error_deg for r in rows})
    lines = ["| axis error | tasks | succeed | median extra grasp force | worst extra grasp force |",
             "|---|---|---|---|---|"]
    for e in errs:
        rs = [r for r in rows if r.added_error_deg == e]
        ex = [r.excess_n for r in rs]
        label = "measured twin" if e == 0 else f"twin + {e:g}°"
        lines.append(f"| {label} | {len(rs)} | {np.mean([r.success for r in rs]):.0%} | "
                     f"{np.median(ex):.0f} N | {np.max(ex):.0f} N |")
    return "\n".join(lines)


def by_kind_table(rows: list[Row]) -> str:
    errs = sorted({r.added_error_deg for r in rows})
    lines = ["| axis error | drawers and slides: extra force | doors and lids: extra force |", "|---|---|---|"]
    for e in errs:
        p = [r.excess_n for r in rows if r.added_error_deg == e and r.kind == "pull"]
        s = [r.excess_n for r in rows if r.added_error_deg == e and r.kind == "swing"]
        f = lambda xs: f"{np.median(xs):.0f} N (max {np.max(xs):.0f})" if xs else "n/a"
        label = "measured twin" if e == 0 else f"twin + {e:g}°"
        lines.append(f"| {label} | {f(p)} | {f(s)} |")
    return "\n".join(lines)


def to_json(rows: list[Row]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=1)
