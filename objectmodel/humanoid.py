"""Twin-to-Task: a humanoid upper body opens real objects using only the measured twin.

The twin (what ObjectModel estimated from a capture) plans the task: where the handle is, which way
the part moves, and the path the hand must follow. The task then runs in a *separate* MuJoCo world
built from the object's ground truth. If the twin's hinge axis or pivot is wrong, the hand drags the
part along a path the real joint cannot follow, and the grasp constraint force shows it.

Robot: a generic 7-DoF-per-arm humanoid upper body on a height-adjustable pedestal (procedural MJCF,
not any company's robot model). Position-controlled joints, damped least-squares IK on the hand site,
gravity compensation. The grasp is a MuJoCo `connect` constraint switched on at contact, which is
how most manipulation benchmarks model a closed gripper. Robot geometry does not collide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import axis_angle_to_matrix, matrix_to_quat_wxyz
from .model import ArticulatedModel, Joint

CTRL_HZ = 100
SIM_DT = 0.002
STANDOFF = 0.52  # metres between handle and torso, horizontally
SHOULDER_ABOVE_HANDLE = 0.28
ARM_KP = 900  # position-servo stiffness, N·m/rad
COMFORT_REACH = 0.45  # metres from shoulder to grip: elbow bent, well inside the 0.66 m arm
SAFE_FORCE_N = 40.0  # peak grasp force above this counts as fighting the joint
OPEN_FRACTION = 0.85  # of the observed range

PHASES = (("reach", 1.6), ("approach", 1.0), ("open", 4.0), ("hold", 0.6))

ARM_JOINTS = (  # name, axis, range (rad), default
    ("sh_pitch", "0 1 0", (-3.0, 1.2), -0.5),
    ("sh_roll", "1 0 0", (-1.6, 1.6), 0.0),
    ("sh_yaw", "0 0 1", (-1.8, 1.8), 0.0),
    ("elbow", "0 1 0", (-2.5, 0.0), -1.2),
    ("wr_yaw", "0 0 1", (-1.8, 1.8), 0.0),
    ("wr_pitch", "0 1 0", (-1.4, 1.4), 0.0),
    ("wr_roll", "1 0 0", (-1.6, 1.6), 0.0),
)


# --------------------------------------------------------------------------- the robot

def _arm_xml(side: str, sign: float) -> str:
    """One 7-DoF arm hanging from the shoulder (robot frame: x forward, y left, z up)."""
    j = {n: f"{side}_{n}" for n, *_ in ARM_JOINTS}
    rng = {n: r for n, _, r, _ in ARM_JOINTS}

    def joint(n: str, axis: str) -> str:
        lo, hi = rng[n]
        if n in ("sh_roll", "sh_yaw", "wr_yaw", "wr_roll") and sign < 0:
            lo, hi = -hi, -lo
        return f'<joint name="{j[n]}" axis="{axis}" range="{lo} {hi}" damping="9" armature="0.05"/>'

    mat = 'rgba="0.80 0.75 0.62 1"'
    dark = 'rgba="0.30 0.34 0.22 1"'
    return f"""
      <body name="{side}_shoulder" pos="0 {sign * 0.21:.3f} 0.42" gravcomp="1">
        {joint("sh_pitch", "0 1 0")}{joint("sh_roll", "1 0 0")}{joint("sh_yaw", "0 0 1")}
        <geom type="sphere" size="0.055" {dark} mass="0.8"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.29" size="0.045" {mat} mass="1.6"/>
        <body name="{side}_forearm" pos="0 0 -0.30" gravcomp="1">
          {joint("elbow", "0 1 0")}
          <geom type="sphere" size="0.045" {dark} mass="0.4"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.26" size="0.038" {mat} mass="1.1"/>
          <body name="{side}_hand" pos="0 0 -0.27" gravcomp="1">
            {joint("wr_yaw", "0 0 1")}{joint("wr_pitch", "0 1 0")}{joint("wr_roll", "1 0 0")}
            <geom type="box" size="0.03 0.045 0.05" pos="0 0 -0.04" {dark} mass="0.4"/>
            <site name="{side}_grip" pos="0 0 -0.09" size="0.012" rgba="0.76 0.57 0.27 1"/>
          </body>
        </body>
      </body>"""


def robot_xml(pos: np.ndarray, yaw: float, pedestal: float) -> str:
    q = matrix_to_quat_wxyz(axis_angle_to_matrix([0, 0, 1], yaw))
    return f"""
    <body name="humanoid" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <geom type="cylinder" fromto="0 0 {-pedestal:.4f} 0 0 0" size="0.06" rgba="0.30 0.34 0.22 1" mass="5"/>
      <geom type="box" pos="0 0 0.25" size="0.11 0.17 0.22" rgba="0.80 0.75 0.62 1" mass="8"/>
      <geom type="box" pos="0 0 0.02" size="0.12 0.15 0.05" rgba="0.30 0.34 0.22 1" mass="2"/>
      <geom type="sphere" pos="0 0 0.62" size="0.1" rgba="0.80 0.75 0.62 1" mass="1.5"/>
      <geom type="box" pos="0.085 0 0.63" size="0.025 0.07 0.03" rgba="0.17 0.20 0.10 1" mass="0.1"/>
      {_arm_xml("r", -1.0)}
      {_arm_xml("l", 1.0)}
    </body>"""


def actuators_xml() -> str:
    rows = []
    for side in ("r", "l"):
        for n, *_ in ARM_JOINTS:
            rows.append(f'<position name="a_{side}_{n}" joint="{side}_{n}" kp="{ARM_KP}" forcerange="-160 160"/>')
    return "<actuator>" + "".join(rows) + "</actuator>"


# --------------------------------------------------------------------------- twin geometry

def _joint_motion(j: Joint, dq: float, pts: np.ndarray) -> np.ndarray:
    if j.type in ("revolute", "continuous"):
        R = axis_angle_to_matrix(j.axis, dq)
        return (pts - j.origin) @ R.T + j.origin
    if j.type == "prismatic":
        return pts + dq * j.axis
    return pts


def _child_box(model: ArticulatedModel, j: Joint, q: float):
    """Child box centre and axes at joint value q (twin kinematics, parent assumed static)."""
    g = model.link(j.child).geometry
    c = _joint_motion(j, q, g.center[None])[0]
    R = g.rotation
    if j.type in ("revolute", "continuous"):
        R = axis_angle_to_matrix(j.axis, q) @ R
    return c, R, g.size


def closed_and_open(model: ArticulatedModel, j: Joint) -> tuple[float, float]:
    """Closed = the observed extreme where the child sits nearest the parent."""
    pc = model.link(j.parent).geometry.center
    d = lambda q: np.linalg.norm(_child_box(model, j, q)[0] - pc)
    qc, qf = (j.lower, j.upper) if d(j.lower) <= d(j.upper) else (j.upper, j.lower)
    return qc, qc + OPEN_FRACTION * (qf - qc)


@dataclass
class Affordance:
    joint: str
    child: str
    kind: str  # pull (prismatic) | swing (revolute)
    handle: np.ndarray  # world point at the closed pose
    normal: np.ndarray  # horizontal unit vector pointing out of the face, toward the robot
    q_closed: float
    q_open: float
    edge: np.ndarray = field(default_factory=lambda: np.zeros(3))  # hinge -> free edge (doors), horizontal


def affordance(model: ArticulatedModel, j: Joint) -> Affordance:
    qc, qo = closed_and_open(model, j)
    c, R, size = _child_box(model, j, qc)
    half = lambda d: float(np.sum(np.abs(R.T @ d) * size / 2))
    pc = model.link(j.parent).geometry.center
    if j.type == "prismatic":
        pull = np.sign(qo - qc) * j.axis
        handle = c + pull * (half(pull) + 0.012)
        normal = pull
        kind = "pull"
    else:
        u = j.axis / np.linalg.norm(j.axis)
        r = c - j.origin
        r = r - np.dot(r, u) * u
        e = r / max(np.linalg.norm(r), 1e-9)  # from hinge toward the free edge
        thin = R[:, int(np.argmin(size))]
        normal = thin if np.dot(thin, c - pc) >= 0 else -thin
        handle = c + e * 0.72 * half(e) + normal * (half(normal) + 0.012)
        kind = "swing"
        edge = e.copy()
    n = normal.copy()
    n[2] = 0.0
    n = n / max(np.linalg.norm(n), 1e-9)
    aff = Affordance(j.name, j.child, kind, handle, n, qc, qo)
    if kind == "swing":
        eh = edge - np.dot(edge, [0, 0, 1]) * np.array([0, 0, 1.0])
        aff.edge = eh / max(np.linalg.norm(eh), 1e-9)
    return aff


def plan_stance(aff: Affordance, path: np.ndarray, robot=None) -> tuple[str, np.ndarray]:
    """Pick the working arm and where its shoulder should be.

    Straight pulls: stand square to the face with the shoulder in line with the handle. Swings:
    grid-search horizontal shoulder positions and keep the one that stays nearest a comfortable
    reach along the whole predicted handle path, without the path running through the torso.
    Robots whose height is fixed (a standing humanoid) search horizontally at their shoulder height.
    """
    left = np.cross([0, 0, 1], -aff.normal)
    travel = float(np.dot(path[-1] - path[0], left))
    side = "l" if travel > 0.05 else "r"
    fixed_z = robot is not None and not robot.adjustable_height
    z = robot.shoulder_z if fixed_z else float(np.mean(path[:, 2])) + SHOULDER_ABOVE_HANDLE
    if aff.kind == "pull" and not fixed_z:
        return "r", aff.handle + aff.normal * STANDOFF + np.array([0, 0, SHOULDER_ABOVE_HANDLE])
    if aff.kind == "pull":
        side = "r"
    reach = getattr(robot, "comfort_reach", COMFORT_REACH)
    best, best_cost = None, np.inf
    clearance = getattr(robot, "body_clearance", 0.15)
    for fwd in np.linspace(clearance, 0.85, 27):
        for lat in np.linspace(-0.45, 0.45, 37):
            if aff.kind == "pull" and abs(lat + 0.0) > 0.06:  # pulls stay square to the face
                continue
            sh = aff.handle + aff.normal * fwd + left * lat
            sh = np.array([sh[0], sh[1], z])
            horiz = np.linalg.norm((path - sh)[:, :2], axis=1)
            if horiz.min() < 0.22:  # the path would run through the torso
                continue
            dist = np.linalg.norm(path - sh, axis=1)
            cost = float(np.max(np.abs(dist - reach)))
            # the working arm should reach to its own side, not across the torso
            sgn = 1.0 if side == "r" else -1.0
            torso = sh + left * getattr(robot, "shoulder_lateral", 0.21) * sgn
            across = float(np.dot(np.mean(path, axis=0) - torso, left)) * sgn
            cost += 0.5 * max(0.0, across) + 1e-3 * abs(lat)  # small tie-break toward standing centred
            if cost < best_cost:
                best, best_cost = sh, cost
    return side, best


def handle_path(model: ArticulatedModel, j: Joint, aff: Affordance, q: float, point: np.ndarray | None = None) -> np.ndarray:
    """Where the twin says a point on the part (the handle by default) is when the joint is at q."""
    p = aff.handle if point is None else point
    return _joint_motion(j, q - aff.q_closed, p[None])[0]


# --------------------------------------------------------------------------- the scene

@dataclass
class TaskResult:
    joint: str
    kind: str
    opened_fraction: float  # of the planned opening, measured on the ground-truth joint
    peak_force_n: float
    mean_force_n: float
    max_track_mm: float
    success: bool
    frames: list = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "frames"}


def _ik_step(mujoco, m, ik, d, q_ref, arm, arm_q, arm_v, site, target, default, iters: int = 3, lam: float = 0.03):
    """Damped least-squares IK on the reference configuration (not the lagging measured one)."""
    jacp = np.zeros((3, m.nv))
    lo, hi = m.jnt_range[arm, 0], m.jnt_range[arm, 1]
    for _ in range(iters):
        ik.qpos[:] = d.qpos
        ik.qpos[arm_q] = q_ref
        mujoco.mj_kinematics(m, ik)
        mujoco.mj_comPos(m, ik)
        err = target - ik.site_xpos[site]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(m, ik, jacp, None, site)
        J = jacp[:, arm_v]
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), err)
        dq += 0.05 * (np.eye(len(arm_v)) - np.linalg.pinv(J) @ J) @ (default - q_ref)
        q_ref = np.clip(q_ref + np.clip(dq, -0.08, 0.08), lo, hi)
    return q_ref


def _smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def execute(world_mjcf: str, world_child: str, world_joint: str, world_q: tuple[float, float],
            twin: ArticulatedModel, twin_joint: Joint, robot=None, lift: float = 0.0,
            render: bool = False, fps: int = 30, size=(960, 540)) -> TaskResult:
    """Plan with the twin, act in the world.

    world_q is (closed, open) for the world's own joint, which may use the opposite sign convention
    from the twin's; it is used only to set the starting state and to score the result.
    lift raises the whole object (on a stand) to a robot's working height.
    """
    import re

    import mujoco

    from .robots import GenericHumanoid

    robot = robot or GenericHumanoid()
    up = np.array([0.0, 0.0, lift])
    aff = affordance(twin, twin_joint)
    aff.handle = aff.handle + up
    path = np.array([handle_path(twin, twin_joint, aff, aff.q_closed + f * (aff.q_open - aff.q_closed)) for f in np.linspace(0, 1, 9)])
    side, shoulder = plan_stance(aff, path, robot)
    left = np.cross([0, 0, 1], -aff.normal)
    sgn = 1.0 if side == "r" else -1.0
    base = shoulder + left * robot.shoulder_lateral * sgn
    base[2] = max(shoulder[2] - robot.shoulder_height, 0.12)
    yaw = float(np.arctan2(-aff.normal[1], -aff.normal[0]))
    world = world_mjcf.replace('contype="1" conaffinity="1"', 'contype="0" conaffinity="0"')
    if lift:
        # put the object on a stand: shift its root body and draw the stand under it
        def shift(mm):
            moved = np.array(mm.group(2).split(), float) + up
            return mm.group(1) + " ".join(f"{v:.5f}" for v in moved) + mm.group(3)

        world = re.sub(r'(<worldbody>.*?<body name="[^"]+" pos=")([^"]+)(")', shift, world, count=1, flags=re.S)
        g = twin.root.geometry
        world = world.replace(
            "  </worldbody>",
            f'    <geom name="stand" type="box" pos="{g.center[0]:.4f} {g.center[1]:.4f} {lift / 2:.4f}" '
            f'size="{g.size[0] / 2:.4f} {g.size[1] / 2:.4f} {lift / 2:.4f}" rgba="0.36 0.40 0.25 1"/>\n  </worldbody>', 1)
    xml = robot.compose(world, base, yaw, world_child, side)
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, world_joint)
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr] = world_q[0]
    pose = robot.default_pose()
    for name, v in pose.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, d)
    for a in range(m.nu):  # every servo holds where it starts
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
    names = robot.ik_joints(side)
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    arm_q = [m.jnt_qposadr[a] for a in arm]
    arm_v = [m.jnt_dofadr[a] for a in arm]
    arm_acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, robot.actuator_for(n)) for n in names]
    default = np.array([pose.get(n, 0.0) for n in names])
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, robot.grip_site(side))
    child = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, world_child)
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    anchor = np.array(robot.anchor(side))
    q_ref = d.qpos[arm_q].copy()
    ik = mujoco.MjData(m)  # kinematic copy: IK runs on the reference pose, the servos track it

    pre = aff.handle + aff.normal * 0.12
    ends = np.cumsum([dur for _, dur in PHASES])
    total = float(ends[-1])
    steps_per_ctrl = int(round(1.0 / (CTRL_HZ * SIM_DT)))
    forces, tracks, frames = [], [], []
    renderer = None
    if render:
        renderer = mujoco.Renderer(m, size[1], size[0])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = 0.5 * aff.handle + 0.5 * np.array([base[0], base[1], aff.handle[2]]) + np.array([0, 0, 0.1])
        cam.distance = 2.5 if robot.adjustable_height else 3.1
        # three-quarter side view from the working arm's side: arm, hand and moving part all visible
        # stand beside the working arm, a little toward the robot, and look across at the part's face
        # (MuJoCo's free camera sits opposite its azimuth direction)
        lateral = -left if side == "r" else left
        v = lateral * 1.0 + aff.normal * 0.45
        cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0])))
        cam.elevation = -14.0
    start_site = d.site_xpos[site].copy()
    grasp_point = aff.handle - up
    t, step, next_frame = 0.0, 0, 0.0
    while t < total:
        if step % steps_per_ctrl == 0:
            if t < ends[0]:
                target = start_site + (pre - start_site) * _smooth(t / ends[0])
            elif t < ends[1]:
                target = pre + (aff.handle - pre) * _smooth((t - ends[0]) / (ends[1] - ends[0]))
            elif t < ends[2]:
                if not d.eq_active[eq]:
                    # close the hand: tie the grip point to wherever it touches the part
                    Rc = d.xmat[child].reshape(3, 3)
                    m.eq_data[eq][0:3] = anchor
                    m.eq_data[eq][3:6] = Rc.T @ (d.site_xpos[site] - d.xpos[child])
                    d.eq_active[eq] = 1
                    grasp_point = d.site_xpos[site].copy() - up  # plan where the hand actually holds
                s_ = _smooth((t - ends[1]) / (ends[2] - ends[1]))
                target = handle_path(twin, twin_joint, aff, aff.q_closed + s_ * (aff.q_open - aff.q_closed), grasp_point) + up
            else:
                target = handle_path(twin, twin_joint, aff, aff.q_open, grasp_point) + up
            q_ref = _ik_step(mujoco, m, ik, d, q_ref, arm, arm_q, arm_v, site, target, default)
            err = target - d.site_xpos[site]
            d.ctrl[arm_acts] = q_ref
            if ends[1] <= t < ends[3] and d.eq_active[eq]:
                rows = np.flatnonzero(d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                forces.append(float(np.linalg.norm(d.efc_force[rows])) if len(rows) else 0.0)
                tracks.append(float(np.linalg.norm(err)))
        mujoco.mj_step(m, d)
        if renderer is not None and t >= next_frame:
            renderer.update_scene(d, cam)
            frames.append(renderer.render().copy())
            next_frame += 1.0 / fps
        t += SIM_DT
        step += 1
    if renderer is not None:
        renderer.close()
    span = world_q[1] - world_q[0]
    opened = (d.qpos[qadr] - world_q[0]) / span if span else 0.0
    peak = max(forces) if forces else 0.0
    res = TaskResult(twin_joint.name, aff.kind, float(opened), peak, float(np.mean(forces)) if forces else 0.0,
                     1000 * max(tracks) if tracks else 0.0, bool(opened >= 0.9 and peak < SAFE_FORCE_N))
    res.frames = frames
    return res
