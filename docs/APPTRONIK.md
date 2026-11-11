# ObjectModel + Twin-to-Task: a portfolio project for Apptronik's software teams

*Independent student project. Not affiliated with, sponsored by, or endorsed by Apptronik. The Apollo
simulation model used here is the public one Google DeepMind publishes in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/apptronik_apollo)
under Apache-2.0. No Apptronik logos, images or proprietary data are used.*

## The one-paragraph pitch

Humanoids that work in factories and warehouses keep meeting things that move: cabinet doors, drawers,
tote and toolbox lids, fixtures. ObjectModel turns a capture of such an object into a measured,
simulator-checked digital twin (URDF + MJCF). **Twin-to-Task** then hands that twin to a humanoid,
the public Apollo model or a generic upper body, and has it open the object in MuJoCo. The task is
*planned only from the twin* and *executed against the object's ground truth*, so the simulator scores
whether the twin was good enough to act on, not just good enough to look at.

## What the robot does

1. **Affordance from the twin.** Find the handle on the moving part, which way it moves (pull or swing),
   which end of the observed range is "closed", and the path the handle follows as the joint opens.
2. **Stance.** Pick the working arm and where to stand. A straight pull stands square to the face. A
   swing grid-searches shoulder positions so the whole handle arc stays at a comfortable reach and never
   sweeps through the torso. A standing robot whose height is fixed searches at its own shoulder height,
   and low furniture goes on a stand because squatting isn't modelled.
3. **Reach, grasp, open, hold.**
   - Damped-least-squares IK runs on the *reference* configuration, in a kinematic copy, while the
     position servos track it. That avoids the windup you get from integrating IK on a lagging
     measured state.
   - On Apollo the IK spans torso yaw, torso pitch and the 7-DoF arm.
   - Both robots get gravity-compensation feedforward. Apollo's model has realistic joint friction,
     so it also gets task-space integral action while reaching, faded out once it has grasped.
   - The hand closes (a MuJoCo `connect` constraint) **only if it is within 3 cm of the real part's
     surface**. A twin that puts the handle in the wrong place misses, and the miss counts as a failure.
   - Once grasped, the hand follows the twin-predicted path of the point it *actually* holds.
4. **Score in the truth.** Recorded: the opening reached on the true joint, the peak grasp force, the
   hand's deviation from its planned path, and whether the grasp happened. The same task is replayed
   with the true joint (the oracle).
   - **Feasible:** the oracle grasps, opens the part at least 90%, and peaks at 120 N or less.
   - **Success:** the twin plan does the same with at most 10 N more peak force than the oracle.
   - Tasks the robot cannot do even with the true joint are reported as infeasible, never as passes.

## Results (reproduce: `objectmodel task-bench`, `objectmodel task fridge --robot apollo`)

Task suite: every first-level door, drawer and lid on 5 objects (10 tasks), generic humanoid.

| Twin used for planning | Tasks that succeed | Median extra grasp force | Worst extra grasp force |
|---|---|---|---|
| ObjectModel's measured twin | **100%** | **0 N** | 3 N |
| Twin + 2° axis error | 60% | 16 N | 35 N |
| Twin + 5° axis error | 30% | 56 N | 225 N |
| Twin + 10° axis error | 0% | 116 N | 823 N |
| Twin + 15° axis error | 0% | 175 N | 1070 N |

ObjectModel's twins land within about 0.2° of the true axis at 1 mm tracking noise (see the main
README benchmark), so they sit comfortably below the ~2° point where tasks start to fail.

On the public Apollo model, all four fridge-door and cabinet tasks succeed with peak grasp forces of
13–18 N, matching the oracle plan.

## How this maps to the roles Apptronik hires for

From public postings (2025–2026, collected Nov 2026; see `research_notes/Apptronik/roles.md`, and check
which are currently open):

| Team / role (as posted) | What they ask for | Where it shows up here |
|---|---|---|
| **Developer Experience SWE intern** | C++/Python/TS, Git, Linux, experience with AI coding agents, tools that help engineers deploy and test | One-command CLI (`objectmodel task`, `bench`, `task-bench`), an MCP server so agents can call the pipeline, a hosted REST API and playground, CI on every push, 125+ tests, a dev server that mirrors production |
| **Robotics SWE intern, real-time controls** | C++, coordinate frames, linear algebra, physics, actuators | A C++17/Eigen core (`cpp/twin_core.cpp`) for Kabsch, joint fitting and the DLS IK step. It matches NumPy to ~1e-16, fits revolute joints ~30× faster than the vectorised NumPy path (benchmarking it also exposed a full-matrix SVD that made the NumPy path 16× slower than needed; fixed), and takes 0.16 µs per IK step. Also: quaternion and RPY conventions checked URDF-vs-MJCF to 0.00 mm, and servo tracking under constraint load |
| **Motion control and planning** | MuJoCo/Isaac/Drake, whole-body control | Upper-body IK across torso and arm on the Apollo model, stance planning against the predicted handle arc, constraint-force metrics |
| **Simulation, core platform, sim-to-real** | MuJoCo or Isaac Lab, CI, "accurate enough for controls validation", system ID | Simulator-in-the-loop validation of every twin (compile, replay, collision sweep, stability), estimating joint parameters from motion (a system-ID cousin), noise and camera-motion sweeps, a reproducible benchmark harness |
| **Autonomy intern** | Python/C++, ROS 2, validate in sim then on hardware | An end-to-end perceive → model → plan → act → verify loop in sim. *Not yet ROS 2: see next steps.* |
| **Manipulation, motion data, teleop** | FK/IK, retargeting, data pipelines | Twins as reusable task assets (handle, path, limits) that a teleop or learning pipeline could consume, plus IK and FK utilities |
| **Perception** | CV, PyTorch | LK tracking with forward-backward checks, depth back-projection, RANSAC rigid segmentation, ArUco metric scale |
| **Fleet Connect (full-stack)** | APIs, Docker/K8s, Python, TypeScript | Python serverless API with input validation and tests, a static web app with live 3D, deploy config |

A one-page PDF version of this brief: [`docs/ObjectModel_Apptronik_brief.pdf`](ObjectModel_Apptronik_brief.pdf).

## Run it

```bash
pip install -e ".[dev]"
objectmodel task kitchen_cabinet                 # generic humanoid opens the door and drawer
objectmodel task kitchen_cabinet --perturb 10    # the same, planned from a twin with a 10° axis error
objectmodel task-bench --out docs/task_bench.md  # the table above

# the public Apollo model (Apache-2.0, Google DeepMind / MuJoCo Menagerie)
bash scripts/fetch_menagerie.sh
MUJOCO_GL=egl objectmodel task fridge --robot apollo --video fridge.mp4
```

## Honest limits and what I would build next

- **Grasping** is a constraint at contact, not a hand-closure controller. Next: use the hand model's
  fingers, or a suction or hook gripper with contact forces.
- **The pelvis is held fixed** (as a balance controller would). There is no locomotion and no squatting.
  Next: couple this to a whole-body controller so the robot can step and crouch to low handles.
- **The robot does not collide** with the object during the task. Next: enable contacts and add a
  collision-aware stance and IK.
- **The control loop runs in Python.** The C++ core already holds the kinematics it needs. Next: move the
  1 kHz loop into C++ and add a ROS 2 node that publishes the twin and the task plan (`/object_model`,
  `/task_plan`), so the same plan drives a sim or a real robot.
- **Captures are synthetic**, with exact ground truth. Next: real RGB-D captures of real cabinets,
  scored against measured hinges.
