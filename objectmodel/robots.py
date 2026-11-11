"""Robots that can perform Twin-to-Task manipulation.

* ``generic``  - a procedural 7-DoF-per-arm humanoid upper body on a height-adjustable pedestal.
  Ships with the package; used by tests, the CLI default and the hosted API.
* ``apollo``   - the Apptronik Apollo model published by Google DeepMind in MuJoCo Menagerie
  (Apache-2.0, https://github.com/google-deepmind/mujoco_menagerie/tree/main/apptronik_apollo).
  Not bundled: set OBJECTMODEL_MENAGERIE to a Menagerie checkout or run scripts/fetch_menagerie.sh.
  The pelvis is held at its standing pose (as a balance controller would) and the upper body
  reaches with torso yaw/pitch plus the 7-DoF arm, all through the model's own position actuators.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .geometry import axis_angle_to_matrix, matrix_to_quat_wxyz

MENAGERIE_ENV = "OBJECTMODEL_MENAGERIE"
DEFAULT_MENAGERIE = os.path.expanduser("~/.cache/objectmodel/mujoco_menagerie")


STUDIO = """<visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/>
    <headlight ambient="0.42 0.42 0.40" diffuse="0.55 0.55 0.52" specular="0.1 0.1 0.1"/></visual>
  <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.92 0.90" rgb2="0.80 0.79 0.75" width="256" height="256"/>
    <texture name="tiles" type="2d" builtin="checker" rgb1="0.86 0.85 0.81" rgb2="0.80 0.79 0.75" width="512" height="512"/>
    <material name="studio_floor" texture="tiles" texrepeat="8 8" reflectance="0.08"/></asset>"""


def _studio_floor(xml: str) -> str:
    return xml.replace('<geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1" pos="0 0 -0.001"/>',
                       '<geom name="floor" type="plane" size="6 6 0.1" material="studio_floor" pos="0 0 -0.001"/>'
                       '<light pos="1.5 -2 3.5" dir="-0.4 0.5 -1" diffuse="0.5 0.5 0.48" castshadow="true"/>')


@dataclass(frozen=True)
class Placement:
    base: np.ndarray
    yaw: float
    side: str


class Robot:
    name = "robot"
    adjustable_height = True
    shoulder_lateral = 0.21  # torso centre to shoulder, metres
    shoulder_height = 0.42  # base frame to shoulder
    shoulder_z = 0.0  # fixed shoulder height for robots that cannot adjust it
    comfort_reach = 0.45
    body_clearance = 0.15  # closest the shoulder may stand to the handle, horizontally

    def ik_joints(self, side: str) -> list[str]:
        raise NotImplementedError

    def actuator_for(self, joint: str) -> str:
        raise NotImplementedError

    def grip_site(self, side: str) -> str:
        return f"{side}_grip"

    def hand_body(self, side: str) -> str:
        raise NotImplementedError

    def anchor(self, side: str) -> tuple[float, float, float]:
        raise NotImplementedError

    def default_pose(self) -> dict[str, float]:
        return {}

    def compose(self, world_mjcf: str, base: np.ndarray, yaw: float, child: str, side: str) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- generic

class GenericHumanoid(Robot):
    name = "generic"

    def ik_joints(self, side: str) -> list[str]:
        from .humanoid import ARM_JOINTS
        return [f"{side}_{n}" for n, *_ in ARM_JOINTS]

    def actuator_for(self, joint: str) -> str:
        return f"a_{joint}"

    def hand_body(self, side: str) -> str:
        return f"{side}_hand"

    def anchor(self, side: str) -> tuple[float, float, float]:
        return (0.0, 0.0, -0.09)

    def default_pose(self) -> dict[str, float]:
        from .humanoid import ARM_JOINTS
        return {f"{s}_{n}": dv for s in ("r", "l") for n, *_r, dv in ARM_JOINTS}

    def compose(self, world_mjcf: str, base: np.ndarray, yaw: float, child: str, side: str) -> str:
        from .humanoid import actuators_xml, robot_xml
        xml = _studio_floor(world_mjcf).replace(
            '<compiler angle="radian" autolimits="true"/>', '<compiler angle="radian" autolimits="true"/>\n  ' + STUDIO)
        xml = xml.replace("  </worldbody>", robot_xml(base, yaw, base[2]) + "\n  </worldbody>")
        a = self.anchor(side)
        eq = (f'<equality><connect name="grasp" body1="{self.hand_body(side)}" body2="{child}" '
              f'anchor="{a[0]} {a[1]} {a[2]}" active="false"/></equality>')
        return xml.replace("</mujoco>", actuators_xml() + eq + "\n</mujoco>")


# --------------------------------------------------------------------------- Apollo (Menagerie)

APOLLO_ARM = ("shoulder_aa", "shoulder_ie", "shoulder_fe", "elbow_fe", "wrist_roll", "wrist_yaw", "wrist_pitch")
# palm centre in the wrist_pitch frame, measured from the hand meshes at the stand pose
APOLLO_PALM = {"r": (0.007, 0.046, -0.122), "l": (0.007, -0.046, -0.122)}
APOLLO_STAND = {  # from the model's "stand" keyframe
    "torso_yaw": 0.0, "torso_roll": 0.0, "torso_pitch": 0.0, "neck_yaw": 0.0, "neck_roll": 0.0, "neck_pitch": 0.0,
    "l_shoulder_aa": 0.2, "l_shoulder_ie": 0.0, "l_shoulder_fe": 0.1, "l_elbow_fe": -0.4,
    "l_wrist_roll": 0.0, "l_wrist_yaw": 0.0, "l_wrist_pitch": 0.0,
    "r_shoulder_aa": -0.2, "r_shoulder_ie": 0.0, "r_shoulder_fe": 0.1, "r_elbow_fe": -0.4,
    "r_wrist_roll": 0.0, "r_wrist_yaw": 0.0, "r_wrist_pitch": 0.0,
    "l_hip_ie": 0.08, "l_hip_aa": 0.1, "l_hip_fe": -0.477, "l_knee_fe": 1.033, "l_ankle_ie": -0.03, "l_ankle_pd": -0.58,
    "r_hip_ie": -0.08, "r_hip_aa": -0.1, "r_hip_fe": -0.477, "r_knee_fe": 1.033, "r_ankle_ie": 0.03, "r_ankle_pd": -0.58,
}
APOLLO_PELVIS_Z = 1.01597  # pelvis height in the stand keyframe (feet on the floor)


def menagerie_path() -> str | None:
    root = os.environ.get(MENAGERIE_ENV, DEFAULT_MENAGERIE)
    xml = os.path.join(root, "apptronik_apollo", "apptronik_apollo.xml")
    return xml if os.path.exists(xml) else None


class ApolloHumanoid(Robot):
    name = "apollo"
    adjustable_height = False
    shoulder_lateral = 0.20
    shoulder_height = 1.367 - APOLLO_PELVIS_Z  # pelvis frame to shoulder
    shoulder_z = 1.367
    comfort_reach = 0.52
    body_clearance = 0.34  # Apollo's chest and arms need room in front of the face

    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or menagerie_path()
        if not self.xml_path:
            raise FileNotFoundError(
                "Apollo model not found. Run scripts/fetch_menagerie.sh or set OBJECTMODEL_MENAGERIE to a "
                "mujoco_menagerie checkout (the model is Apache-2.0, published by Google DeepMind).")

    def ik_joints(self, side: str) -> list[str]:
        # whole upper body: lean and twist the torso, then the 7-DoF arm
        return ["torso_yaw", "torso_pitch"] + [f"{side}_{n}" for n in APOLLO_ARM]

    def actuator_for(self, joint: str) -> str:
        return joint  # the Menagerie model names each position actuator after its joint

    def hand_body(self, side: str) -> str:
        return f"{side}_wrist_pitch_link"

    def anchor(self, side: str) -> tuple[float, float, float]:
        return APOLLO_PALM[side]

    def default_pose(self) -> dict[str, float]:
        pose = dict(APOLLO_STAND)
        for s, sign in (("r", -1), ("l", 1)):  # arms forward and bent, ready to reach
            pose.update({f"{s}_shoulder_fe": -0.6, f"{s}_elbow_fe": -1.1, f"{s}_shoulder_aa": 0.12 * sign})
        return pose

    def compose(self, world_mjcf: str, base: np.ndarray, yaw: float, child: str, side: str) -> str:
        src = open(self.xml_path).read()
        asset_dir = os.path.join(os.path.dirname(self.xml_path), "assets")
        src = src.replace('meshdir="assets"', f'meshdir="{asset_dir}"').replace('texturedir="assets"', f'texturedir="{asset_dir}"')
        src = re.sub(r'\s*<joint name="floating_base"[^>]*/>', "", src)  # pelvis held at its standing pose
        src = re.sub(r"<keyframe>.*?</keyframe>", "", src, flags=re.S)
        q = matrix_to_quat_wxyz(axis_angle_to_matrix([0, 0, 1], yaw))
        src = re.sub(r'<body name="base_link" pos="[^"]*">',
                     f'<body name="base_link" pos="{base[0]:.4f} {base[1]:.4f} {APOLLO_PELVIS_Z:.5f}" '
                     f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">', src, count=1)
        for s in ("r", "l"):
            p = APOLLO_PALM[s]
            src = re.sub(rf'(<body name="{s}_wrist_pitch_link"[^>]*>)',
                         rf'\1<site name="{s}_grip" pos="{p[0]} {p[1]} {p[2]}" size="0.012" rgba="0.76 0.57 0.27 1"/>',
                         src, count=1)
        world = re.search(r"<worldbody>(.*?)</worldbody>", _studio_floor(world_mjcf), flags=re.S).group(1)
        src = src.replace("</worldbody>", world + "\n  </worldbody>", 1)
        a = self.anchor(side)
        extra = (STUDIO +
                 f'<equality><connect name="grasp" body1="{self.hand_body(side)}" body2="{child}" '
                 f'anchor="{a[0]} {a[1]} {a[2]}" active="false"/></equality>')
        return src.replace("</mujoco>", extra + "\n</mujoco>")


def get_robot(name: str = "generic") -> Robot:
    if name == "generic":
        return GenericHumanoid()
    if name == "apollo":
        return ApolloHumanoid()
    raise ValueError(f"unknown robot {name!r}; choose generic or apollo")
