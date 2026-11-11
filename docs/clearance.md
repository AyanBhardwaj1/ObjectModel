# Robot–object clearance

The task world turns robot–object contacts off, so nothing physically stops an arm from passing through
the object. To make that visible, every run samples (at 20 Hz) the signed distance between the robot's
collision geometry and the object's, with MuJoCo's `mj_geomDistance`. The hand that holds the handle is
excluded, since it touches the part by design (generic: the hand body; Apollo: the palm and its three
wrist links, which sit together at the wrist; the last column counts the wrist roll link as body). Negative means overlap. Measured twin, no added error,
reproduce with `TaskResult.min_clearance_mm` (also printed by `objectmodel task`).

| object / part | generic (hand excluded) | Apollo before the fix: idle arm forward | Apollo now: palm and whole wrist excluded | Apollo now: wrist roll link counted |
|---|---|---|---|---|
| kitchen_cabinet / door | -45 mm (l_forearm) | -146 mm (r_wrist_pitch_link) | +117 mm (l_hip_fe_link) | -117 mm (l_wrist_roll_link) |
| kitchen_cabinet / drawer | +30 mm (l_hand) | -131 mm (l_wrist_pitch_link) | +135 mm (l_wrist_pitch_link) | -38 mm (r_wrist_roll_link) |
| fridge / right_door | -34 mm (r_forearm) | -236 mm (l_wrist_pitch_link) | +45 mm (r_hip_fe_link) | -102 mm (r_wrist_roll_link) |
| fridge / left_door | -33 mm (l_forearm) | -201 mm (r_wrist_pitch_link) | +44 mm (l_knee_fe_link) | -65 mm (l_wrist_roll_link) |
| dresser / door | -51 mm (l_forearm) | -231 mm (r_wrist_pitch_link) | +41 mm (r_wrist_pitch_link) | -93 mm (l_wrist_roll_link) |
| dresser / drawer1 | +40 mm (l_hand) | -124 mm (l_wrist_pitch_link) | +137 mm (r_hip_fe_link) | -19 mm (r_wrist_roll_link) |
| dresser / drawer0 | +40 mm (l_hand) | -125 mm (l_wrist_pitch_link) | +138 mm (r_hip_fe_link) | -19 mm (r_wrist_roll_link) |
| dresser / drawer2 | +40 mm (l_hand) | -122 mm (l_wrist_pitch_link) | +136 mm (r_hip_fe_link) | +15 mm (r_wrist_roll_link) |
| drawer_cabinet / drawer | +79 mm (r_forearm) | -65 mm (l_wrist_pitch_link) | +140 mm (r_hip_fe_link) | +6 mm (r_wrist_roll_link) |
| laptop / lid | -11 mm (r_forearm) | +31 mm (r_wrist_roll_link) | +109 mm (r_hip_fe_link) | +31 mm (r_wrist_roll_link) |

What this found and what changed:

- **Apollo's idle arm swung into the object** (up to 24 cm of overlap). Both arms started forward and bent;
  when the torso yawed toward the task, the idle forearm went into the cabinet or fridge. The idle arm now
  keeps the model's own stand pose at its side, and all ten tasks still succeed.
- **With the hand and wrist excluded, Apollo stays at least 41 mm from every object** in all ten tasks; the
  closest parts are its legs next to the fridge.
- **Still open: the working wrist or forearm cuts into swinging doors** (Apollo's wrist up to 12 cm; the
  generic forearm up to 5 cm). The IK controls the hand's position only, so the forearm is free to lie
  along the door as it swings. Drawers are fine.
- **Tried and reverted: a soft forearm-direction task in the IK.** Two extra DLS rows (weight 0.5) steered
  the forearm's horizontal direction into the part's face, turning with the planned joint value. On the
  generic arm the doors still overlapped by 47–58 mm, and the cabinet door became infeasible (277 N peak,
  76% open) because the forearm task fought the hand-position task while the door swung toward the
  robot. It did clear the laptop lid (+9 mm from −11 mm). A real fix needs a different grasp for doors
  (hook the free edge, or rotate the hand around the handle with a wrist-orientation task), then
  robot–object contacts on.
