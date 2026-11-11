"""Writers for URDF, MJCF and a JSON scene (used by the web viewer)."""
from __future__ import annotations

import json
from xml.sax.saxutils import quoteattr

import numpy as np

from .geometry import matrix_to_quat_wxyz, matrix_to_rpy
from .model import ArticulatedModel, Link

EFFORT = 10.0  # N or N·m; URDF requires an effort limit
VELOCITY = 2.0


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _parent_origin(model: ArticulatedModel, link: Link) -> np.ndarray:
    for j in model.joints:
        if j.child == link.name:
            return model.link(j.parent).origin
    return np.zeros(3)


def to_urdf(model: ArticulatedModel) -> str:
    out = ['<?xml version="1.0"?>', f"<robot name={quoteattr(model.name)}>"]
    for link in model.links:
        g, ph = link.geometry, link.physics
        local = g.center - link.origin
        rpy = matrix_to_rpy(g.rotation)
        out += [
            f"  <link name={quoteattr(link.name)}>",
            "    <inertial>",
            f'      <origin xyz="{_f(local)}" rpy="{_f(rpy)}"/>',
            f'      <mass value="{ph.mass:.6g}"/>',
            f'      <inertia ixx="{ph.inertia_diag[0]:.6g}" ixy="0" ixz="0" iyy="{ph.inertia_diag[1]:.6g}" iyz="0" izz="{ph.inertia_diag[2]:.6g}"/>',
            "    </inertial>",
        ]
        for tag in ("visual", "collision"):
            out += [
                f"    <{tag}>",
                f'      <origin xyz="{_f(local)}" rpy="{_f(rpy)}"/>',
                f'      <geometry><box size="{_f(g.size)}"/></geometry>',
                f"    </{tag}>",
            ]
        out.append("  </link>")
    for j in model.joints:
        rel = model.link(j.child).origin - model.link(j.parent).origin
        out += [
            f"  <joint name={quoteattr(j.name)} type={quoteattr(j.type)}>",
            f"    <parent link={quoteattr(j.parent)}/>",
            f"    <child link={quoteattr(j.child)}/>",
            f'    <origin xyz="{_f(rel)}" rpy="0 0 0"/>',
        ]
        if j.type != "fixed":
            out.append(f'    <axis xyz="{_f(j.axis)}"/>')
            out.append(f'    <dynamics damping="{j.damping:.6g}" friction="{j.friction:.6g}"/>')
        if j.type in ("revolute", "prismatic"):
            out.append(f'    <limit lower="{j.lower:.6g}" upper="{j.upper:.6g}" effort="{EFFORT}" velocity="{VELOCITY}"/>')
        out.append("  </joint>")
    out.append("</robot>")
    return "\n".join(out) + "\n"


MJ_TYPE = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide"}
PALETTE = ["0.85 0.55 0.3 1", "0.3 0.6 0.85 1", "0.45 0.75 0.4 1", "0.8 0.4 0.6 1", "0.6 0.5 0.85 1", "0.9 0.8 0.3 1"]


def to_mjcf(model: ArticulatedModel) -> str:
    out = [
        f"<mujoco model={quoteattr(model.name)}>",
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option gravity="0 0 -9.81" timestep="0.002"/>',
        "  <worldbody>",
        '    <light pos="0 0 3" dir="0 0 -1"/>',
        '    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1" pos="0 0 -0.001"/>',
    ]

    def body(link: Link, depth: int, idx: list[int]) -> None:
        pad = "  " * depth
        pos = link.origin - _parent_origin(model, link) if depth > 2 else link.origin
        g, ph = link.geometry, link.physics
        q = matrix_to_quat_wxyz(g.rotation)
        local = g.center - link.origin
        out.append(f'{pad}<body name={quoteattr(link.name)} pos="{_f(pos)}">')
        j = next((j for j in model.joints if j.child == link.name), None)
        if j is not None and j.type in MJ_TYPE:
            rng = "" if j.type == "continuous" else f' range="{j.lower:.6g} {j.upper:.6g}"'
            out.append(
                f'{pad}  <joint name={quoteattr(j.name)} type="{MJ_TYPE[j.type]}" axis="{_f(j.axis)}" '
                f'pos="0 0 0"{rng} damping="{j.damping:.6g}" frictionloss="{j.friction:.6g}"/>'
            )
        out.append(
            f'{pad}  <inertial pos="{_f(local)}" quat="{_f(q)}" mass="{ph.mass:.6g}" diaginertia="{_f(ph.inertia_diag)}"/>'
        )
        out.append(
            f'{pad}  <geom type="box" size="{_f(g.size / 2)}" pos="{_f(local)}" quat="{_f(q)}" '
            f'rgba="{PALETTE[idx[0] % len(PALETTE)]}" contype="1" conaffinity="1"/>'
        )
        idx[0] += 1
        for cj in model.children(link.name):
            body(model.link(cj.child), depth + 1, idx)
        out.append(f"{pad}</body>")

    body(model.root, 2, [0])
    out += ["  </worldbody>", "</mujoco>"]
    return "\n".join(out) + "\n"


def _num(x: float, nd: int):
    """JSON-safe rounding: non-finite values become null."""
    return round(float(x), nd) if np.isfinite(x) else None


def to_scene_json(model: ArticulatedModel, max_points: int = 400) -> dict:
    links = []
    for k, l in enumerate(model.links):
        pts = l.points[:: max(1, len(l.points) // max_points)]
        links.append({
            "name": l.name,
            "origin": l.origin.round(5).tolist(),
            "box": {
                "center": l.geometry.center.round(5).tolist(),
                "rotation": l.geometry.rotation.round(6).tolist(),
                "size": l.geometry.size.round(5).tolist(),
            },
            "material": l.physics.material,
            "mass": round(l.physics.mass, 4),
            "color": PALETTE[k % len(PALETTE)],
            "points": pts.round(4).tolist(),
        })
    joints = [{
        "name": j.name, "type": j.type, "parent": j.parent, "child": j.child,
        "axis": j.axis.round(6).tolist(), "origin": j.origin.round(5).tolist(),
        "lower": j.lower, "upper": j.upper, "confidence": round(j.confidence, 4),
        "axis_std_deg": _num(j.axis_std_deg, 3), "origin_std_mm": _num(j.origin_std * 1000, 2),
        "rms_mm": round(j.rms * 1000, 3), "states": np.round(j.states, 5).tolist(),
    } for j in model.joints]
    return {"name": model.name, "root": model.root.name, "links": links, "joints": joints, "notes": model.notes}


def scene_json_str(model: ArticulatedModel) -> str:
    return json.dumps(to_scene_json(model))
