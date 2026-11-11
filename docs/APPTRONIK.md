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

## If you have five minutes

1. Watch the clip at the top of the Humanoids page (`web/humanoids.html`, or `python scripts/dev_server.py 8000`
   and open `/humanoids`): Apollo plans from a measured twin and swings a fridge door open.
2. Read the results tables below: they show how accurate a twin must be before a robot can act on it.
3. Run `objectmodel task kitchen_cabinet --perturb 5` and compare with the run without `--perturb`. Same
   robot and same object, only the twin differs.
4. Skim `objectmodel/humanoid.py` (planning and control) and `cpp/twin_core.cpp` (the C++ core).
5. For how the project handles its own failures, read `docs/clearance.md`: a check that found a collision
   the simulator was hiding, the fix, and an attempted fix that didn't work.

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
   - Both robots get MuJoCo's ideal body gravity compensation (`gravcomp`), not a modelled feedforward. Apollo's model has realistic joint friction,
     so it also gets task-space integral action while reaching, faded out once it has grasped.
   - The grasp (a MuJoCo `connect` constraint at the hand's current position) engages **only if the hand
     is within 3 cm of the real part's surface**; in the generic website runs it engaged about 2 cm out. A twin that puts the handle in the wrong place misses, and the miss counts as a failure.
   - Once grasped, the hand follows the twin-predicted path of the point it *actually* holds.
4. **Score in the truth.** Recorded: the opening reached on the true joint, the peak grasp force, the
   hand's deviation from its planned path, and whether the grasp happened. The same task is replayed
   with the true joint (the oracle).
   - **Feasible:** the oracle grasps, reaches at least 90% of the planned opening (85% of the joint's observed range), and peaks at 120 N or less.
   - **Success:** the twin plan does the same with at most 10 N more peak force than the oracle.
   - Tasks the robot cannot do even with the true joint are reported as infeasible, never as passes.

## Results (reproduce: `objectmodel task-bench`, `objectmodel task fridge --robot apollo`)

Task suite: every first-level door, drawer and lid on 5 objects, generic humanoid (stiff position-
controlled arm). 7 of 10 tasks are feasible. On the dresser and fridge doors the stiff arm can't follow the wide arc
closely enough even with the true joint (416–615 N peak, about 50 mm tracking error; Apollo opens the
same doors at 45–50 N), so they are reported as infeasible rather than passed.

| Twin used for planning | Tasks that succeed | Median extra grasp force | Worst extra grasp force | Grasp missed |
|---|---|---|---|---|
| ObjectModel's measured twin | **100%** | **+0 N** | +2 N | 0 |
| Twin + 2° axis error | 0% | +19 N | +24 N | 0 |
| Twin + 5° axis error | 0% | +50 N | +77 N | 0 |
| Twin + 10° axis error | 0% | +79 N | +120 N | 5 |
| Twin + 15° axis error | 0% | +125 N | +181 N | 5 |

On a stiff arm even a 2° axis error makes the arm fight the joint. At 10° the twin also places the
drawer handles off the drawer fronts, so the hand misses; the 10° and 15° force figures come from the
2 tasks that still grasped.

ObjectModel's twins average 0.18° of axis error at 1 mm tracking noise (main README benchmark; at most
0.35° in these task runs), but 0.7° at 3 mm noise. That is inside the stiff arm's tolerance, though not
by a wide margin: failures begin between 0.5° and 1°. Noisier captures need a more compliant controller.

On the public Apollo model (`objectmodel task-bench --robot apollo`), all 10 tasks are feasible.
Its joints are compliant, so a wrong twin shows up as hand-path deviation rather than force:

| Twin used for planning | Tasks that succeed | Grasp missed | Median extra hand deviation | Worst extra grasp force |
|---|---|---|---|---|
| ObjectModel's measured twin | **100%** | 0 | +0 mm | +6 N |
| Twin + 1° | 100% | 0 | +0 mm | +5 N |
| Twin + 2° | 100% | 0 | −1 mm | +2 N |
| Twin + 5° | 100% | 0 | +4 mm | +5 N |
| Twin + 10° | 80% | 2 | +22 mm | +7 N |
| Twin + 15° | 60% | 2 | +42 mm | +23 N |

A finer sweep on the stiff arm (`docs/task_bench.md`) puts the threshold near 1°: every degree of axis
error costs about 9 N of extra grasp force, all 7 tasks pass at +0.5°, 4 of 7 at +1° and 1 of 7 at +1.5°.

Takeaway: how accurate a twin must be depends on the controller. A stiff arm needs it within ~1°; a
compliant one tolerates ~5° but pays in path error, which matters near obstacles. Force alone is not a
complete twin-quality metric, so the bench reports both.

## How this maps to the roles Apptronik hires for

From public postings (2025–2026, collected Nov 2026; see `research_notes/Apptronik/roles.md`, and check
which are currently open):

| Team / role (as posted) | What they ask for | Where it shows up here |
|---|---|---|
| **Developer Experience SWE intern** | C++/Python/TS, Git, Linux, experience with AI coding agents, tools that help engineers deploy and test | One-command CLI (`objectmodel task`, `plan`, `bench`, `task-bench`), an MCP server so agents can reconstruct, plan and run humanoid tasks, a hosted REST API and playground, CI on every push, 142 tests, a dev server that mirrors production |
| **Robotics SWE intern, real-time controls** | C++, coordinate frames, linear algebra, physics, actuators | A C++17/Eigen core (`cpp/twin_core.cpp`) for Kabsch, joint fitting and the DLS IK step. It matches NumPy to ~1e-16, fits revolute joints ~30× faster than the vectorised NumPy path (benchmarking it also exposed a full-matrix SVD that made the NumPy path 16× slower than needed; fixed), and does one damped-least-squares solve (3×9 Jacobian) in 0.16 µs, timed inside C++; the control loop itself still runs in Python with MuJoCo Jacobians. Also: quaternion and RPY conventions checked URDF-vs-MJCF to 0.00 mm, and servo tracking under constraint load |
| **Motion control and planning** | MuJoCo/Isaac/Drake, whole-body control | Upper-body IK across torso and arm on the Apollo model, stance planning against the predicted handle arc, constraint-force metrics |
| **Simulation, core platform, sim-to-real** | MuJoCo or Isaac Lab, CI, "accurate enough for controls validation", system ID | Simulator-in-the-loop validation of every twin (compile, replay, collision sweep, stability), estimating joint parameters from motion (a system-ID cousin), noise and camera-motion sweeps, a reproducible benchmark harness, and a robot–object clearance check that caught Apollo's idle arm swinging into the object (`docs/clearance.md`) |
| **Autonomy intern** | Python/C++, ROS 2, validate in sim then on hardware | An end-to-end perceive → model → plan → act → verify loop in sim. *Not yet ROS 2: see next steps.* |
| **Manipulation, motion data, teleop** | FK/IK, retargeting, data pipelines | Twins as reusable task assets: `objectmodel plan <object>` exports a versioned JSON plan per part (handle, approach direction, joint limits, stance, arm and the handle path along the joint) for a teleop, learning or ROS 2 stack to consume. Plus IK and FK utilities |
| **Perception** | CV, PyTorch | LK tracking with forward-backward checks, depth back-projection, RANSAC rigid segmentation, ArUco metric scale |
| **Fleet Connect (full-stack)** | APIs, Docker/K8s, Python, TypeScript | Python serverless API with input validation and tests, a static web app with live 3D, deploy config |

A one-page PDF version of this brief: [`docs/ObjectModel_Apptronik_brief.pdf`](ObjectModel_Apptronik_brief.pdf).

## Run it

```bash
pip install -e ".[dev]"
objectmodel task kitchen_cabinet                 # generic humanoid opens the door and drawer
objectmodel task kitchen_cabinet --perturb 10    # the same, planned from a twin with a 10° axis error
objectmodel task-bench --out docs/task_bench.md  # the table above (overwrites the file)
objectmodel task-bench --errors 0 0.5 1 1.5       # the fine sweep near the threshold
objectmodel plan kitchen_cabinet --out plan.json # the plan as JSON, for another stack

# the public Apollo model (Apache-2.0, Google DeepMind / MuJoCo Menagerie)
bash scripts/fetch_menagerie.sh
MUJOCO_GL=egl objectmodel task fridge --robot apollo --video fridge.mp4
```

## Honest limits and what I would build next

- **Grasping** is a constraint at contact, not a hand-closure controller. Next: use the hand model's
  fingers, or a suction or hook gripper with contact forces.
- **The pelvis is held fixed** (as a balance controller would). There is no locomotion and no squatting.
  Next: couple this to a whole-body controller so the robot can step and crouch to low handles.
- **Robot–object contacts are off** during the task, so the run *measures* clearance instead
  (`docs/clearance.md`). That check found Apollo's idle arm swinging up to 24 cm into the object when
  the torso turned; the idle arm now stays at its side, and the rest of the body keeps at least 41 mm
  clear in all ten tasks. Still open: the working wrist or forearm cuts into swinging doors (up to
  12 cm on Apollo, 5 cm on the generic arm) because the IK controls hand position, not orientation.
  Next: orientation-aware IK, then contacts on.
- **The control loop runs in Python.** The C++ core has the DLS solve, but FK and Jacobians still come from MuJoCo. Next: move the
  1 kHz loop into C++ and add a ROS 2 node that publishes the twin and the `objectmodel plan` JSON as messages (`/object_model`,
  `/task_plan`), so the same plan drives a sim or a real robot.
- **Captures are synthetic**, with exact ground truth. Next: real RGB-D captures of real cabinets,
  scored against measured hinges.
