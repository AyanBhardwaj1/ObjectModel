# Sim Export and Validation of Articulated Models (URDF / MJCF / USD)

Note on sourcing: wiki.ros.org, mujoco.readthedocs.io and arxiv.org could not be fetched from this session (blocked by the session's network settings). MuJoCo facts come from the official doc source on GitHub (raw XMLreference.rst). URDF and paper facts come from search-result summaries of the primary pages. Items marked "general knowledge" are standard engineering facts that I could not verify against a fetched source.

## 1. URDF / MJCF structure for joints, inertia and geometry

### Takeaway
URDF puts each joint at the child link's frame, places the axis in that joint frame, and requires `<limit>` on revolute and prismatic joints. MJCF instead nests the joint inside the child body, with `pos` and `axis` given in the body frame. MJCF angles default to **degrees**. Most simulator failures come from zero or missing mass and inertia, an inertia tensor that is not positive-definite or breaks the triangle inequality, missing limits, and overlapping collision geometry.

### Cited Findings
- **URDF joint types** are revolute, continuous, prismatic, fixed, floating and planar. `type`, `parent` and `child` are required. `<limit lower upper effort velocity>` is required for revolute and prismatic joints, in radians or metres, and is omitted for continuous joints. `<axis>` is optional, defaults to x (1 0 0), is expressed in the joint frame, and is not used by fixed or floating joints. `<origin>` is the transform from the parent link frame to the joint frame and defaults to identity. — [ROS URDF tutorial (docs.ros.org)](https://docs.ros.org/en/rolling/Tutorials/URDF/Building-a-Movable-Robot-Model-with-URDF.html); [MathWorks URDF Primer](https://au.mathworks.com/help/sm/ug/urdf-model-import.html); [urdf-rs parser](https://docs.rs/urdf-rs/0.6.6/src/urdf_rs/deserialize.rs.html)
- **MJCF joint**: `type` is hinge (the default), slide, ball or free. `pos` is "specified in the frame of the body where the joint is defined". `axis` defaults to "0 0 1" and is normalized. Hinge and ball `range` uses the compiler's `angle` unit. `limited="auto"` turns limits on whenever `range` is set. `damping` gives f = −a·v. `frictionloss` models dry friction. `armature` is extra joint-space inertia. `stiffness` and `springref` model springs, and `ref` is the reference position. — [MuJoCo XMLreference.rst](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)
- **MJCF compiler settings**: `angle` defaults to "degree" in MJCF. With `autolimits="true"` (the default), a joint that has a range is limited. When autolimits is false, a joint is only limited if it has both `limited="true"` and a range. `boundmass` and `boundinertia` set lower bounds on body mass and diagonal inertia. `balanceinertia` exists because "a valid diagonal inertia matrix must satisfy A+B>=C for all permutations". — [MuJoCo XMLreference.rst](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)
- **MJCF `<inertial>`**: `pos` is required. Mass cannot be negative. If `fullinertia` (Ixx Iyy Izz Ixy Ixz Iyz) is not positive-definite, compilation fails "unless boundinertia is positive". Geom `density` defaults to 1000 (water) and is ignored when `mass` is set. `contype` and `conaffinity` (both default 1) are collision bitmasks. Geom `group` affects inertia inference through `inertiagrouprange`. — [MuJoCo XMLreference.rst](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)
- **Mesh inertia** can be computed as `convex`, `exact`, `legacy` (the default, which over-counts volume for non-convex meshes) or `shell` (mass on the surface). The docs recommend `convex`. — [MuJoCo XMLreference.rst](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)
- **Isaac Sim URDF importer**: a link with no inertia tensor gets the identity tensor, and a link with no mass gets the "Default Density". Collision can be imported as one Convex Hull or as a Convex Decomposition into several hulls. NVIDIA's troubleshooting guide names overlapping colliders and unrealistic mass or inertia as causes of abnormal forces. — [Isaac Sim URDF Importer](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html); [Robot Setup Troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/troubleshooting.html)
- A forum report says one Isaac Sim version merged collision meshes of links joined by fixed joints into a single convex hull, so collision shapes should be checked after import. — [NVIDIA forum](https://forums.developer.nvidia.com/t/urdf-import-creates-incorrect-collision-geometry-in-isaac-sim-4-5/329945)

### Inferences
- **Exporter checklist (general knowledge combined with the above):**
  - Write SI units and radians everywhere. In MJCF, set `<compiler angle="radian" autolimits="true"/>` explicitly.
  - Give every non-world link mass > 0, a diagonal inertia > 0 that satisfies the triangle inequality, and an explicit inertial origin at the part's centre of mass.
  - Normalize joint axes before writing them.
  - Write `<limit>` for every revolute and prismatic joint, and use `continuous` (URDF) or an unlimited hinge (MJCF) only for truly unbounded rotation.
  - Use simplified convex collision geometry (primitives or a convex decomposition), keep the visual mesh separate, and exclude parent–child contacts. In MJCF, adjacent bodies are excluded by default.
- **Static (fixed) base**: in URDF, attach the base to a `world` link with a fixed joint. In MJCF, simply put no free joint on the base body.

### Gaps
- I could not fetch the full ROS wiki text for `<dynamics damping friction>` or the definition of the joint frame.
- I did not verify USD/PhysX-specific attributes such as `PhysicsRevoluteJoint`, `localPos0/1` and drive APIs.

## 2. Physical property estimation (mass, inertia, materials)

### Takeaway
Use mass = volume × density with a material lookup table. For thin shells such as cabinet panels, use the shell volume (area × thickness) rather than the enclosed volume. Compute inertia from primitive formulas. VLM/LLM methods pick a material or density per part, and for mass they integrate density over a thickness-aware volume.

### Cited Findings
- **Typical densities (kg/m³)**: pine/softwood about 500 (range 370–650); aluminium 2700; ABS about 1050; carbon steel 7850 and stainless 7900–8000; window glass about 2500. — [Engineering Toolbox](https://engineeringtoolbox.com/density-solids-d_1265.html); [AIMS density chart](https://aimsindustrial.com.au/blogs/product-guides/material-density-chart)
- **Solid cylinder** (mass m, radius r, height h, axis z): Iz = ½mr² and Ix = Iy = (1/12)m(3r² + h²). — [Wikipedia: List of moments of inertia](https://en.wikipedia.org/wiki/List_of_moments_of_inertia); [MathWorld](https://mathworld.wolfram.com/MomentofInertia.html)
- **Solid box** (sides a, b, c along x, y, z): Ixx = m(b² + c²)/12, Iyy = m(a² + c²)/12, Izz = m(a² + b²)/12 (general knowledge, listed on [Wikipedia](https://en.wikipedia.org/wiki/List_of_moments_of_inertia)).
- **NeRF2Physics** builds a language-embedded point cloud, has an LLM propose candidate materials, and estimates per-point properties with zero-shot kernel regression. For mass, it integrates density over surface cuboids using thickness estimated by the LLM. It beats zero-shot and supervised baselines for mass on the ABO dataset. — [NeRF2Physics project page](https://ajzhai.github.io/NeRF2Physics)
- **PhysGaussian** requires material parameters to be configured by hand for each scene. Phys4DGen replaces that with "MLLM-guided material identification". — [Phys4DGen](https://arxiv.org/html/2411.16800v5)
- **Other VLM-based approaches**: SOPHY combines VLM guidance with feedback from expert engineers. VoMP combines part-segmented assets, material databases and a VLM. Density is noted to be much harder to estimate than volume. — [SOPHY](https://arxiv.org/html/2504.12684v3); [XDen-1K](https://arxiv.org/html/2512.10668v1)
- **PhysObjects** holds 417K automated plus 39.6K crowd-sourced annotations of physical concepts such as mass and material, with density held out. — [PG-VLM](https://iliad.stanford.edu/pg-vlm/)

### Inferences
- **Practical recipe**:
  1. Have a VLM output `{material, confidence}` per part.
  2. Look up the density from a fixed table rather than asking the LLM for a number.
  3. Compute volume from a watertight mesh (MuJoCo `inertia="exact"`) or from fitted primitives. For hollow parts use a shell with a thickness of roughly 5–20 mm.
  4. Clamp the result to a plausible mass range for the object category.
  5. Inertia errors matter less than joint damping and friction for kinematic replay, but they must stay valid (positive-definite).

### Gaps
- I found no standard benchmark numbers for LLM-only density estimation accuracy.
- I did not verify NeRF2Physics' exact aggregation formula or error values.

## 3. Validation metrics in the articulated-reconstruction literature

### Takeaway
The de facto protocol, set by Ditto and PARIS and reused widely, is:
- **Axis Ang**: angle between the predicted and true axis, in degrees, 0–90°.
- **Axis Pos**: minimum line-to-line distance, revolute joints only.
- **Part Motion / joint state**: geodesic rotation error in degrees for revolute joints, translation error for prismatic joints.
- **Chamfer-L1** on the whole object, static part and movable part.
- **Joint type accuracy.**

### Cited Findings
- **Ditto** reports Chamfer-L1 for part geometry, axis orientation error for all joints, and axis position error ("minimum distance between predicted and GT rotation axis") for revolute joints. On Shape2Motion its joint-type accuracy is 100%. — [Ditto](https://arxiv.org/pdf/2202.08227); [Ditto supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Jiang_Ditto_Building_Digital_CVPR_2022_supplemental.pdf)
- **PARIS** reports:
  - Axis Ang over 0–90°.
  - Axis Pos for revolute joints only.
  - Part Motion as geodesic distance in degrees (revolute) or translation error (prismatic).
  - CD-w, CD-s and CD-m, computed from 10k sampled points in both directions and averaged.
  
  — [PARIS](https://arxiv.org/pdf/2308.07391)
- **Real2Code** reports Chamfer distance plus joint type, axis and position accuracy on PartNet-Mobility, and found that regressing continuous joint parameters gave larger position errors. — [Real2Code](https://arxiv.org/pdf/2406.08474)
- **Articulate-Anything** reports joint Type (fraction wrong), Axis (radians) and Origin (metres). Success on PartNet-Mobility rose to 75%, against 8.7–12.2% for earlier work. Its VLM critic, which compares renders of the simulated URDF with the input video, overestimates success on subtle errors. — [Articulate-Anything](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf); [review](https://liner.com/review/articulateanything-automatic-modeling-of-articulated-objects-via-a-visionlanguage-foundation)
- **Critique**: scoring type, axis, origin and limits independently ignores that together they describe one motion, and that different parameterizations can describe the same motion. — [URDF-Anything+ / related](https://arxiv.org/pdf/2603.14010)

### Inferences
- **Implementation**:
  - Axis Ang: `acos(|â·a|)`. Using the absolute value folds the result into 0–90°.
  - Axis Pos: distance between skew lines, `|(p̂−p)·(â×a)|/‖â×a‖`, falling back to point-to-line distance when the axes are parallel.
  - Revolute state error: `|θ̂−θ|` after aligning the axis sign.
  - Prismatic state error: `|d̂−d|` in metres.

### Gaps
- I could not confirm the exact success thresholds Articulate-Anything uses.
- I could not confirm whether Real2Code computes Axis Pos as a line-to-line distance.

## 4. Meaningful MuJoCo playback validation

### Takeaway
Kinematic replay checks consistency: set qpos to the estimated joint state for each frame, run forward kinematics, and compare each part's simulated pose or rendered silhouette with the observation. Dynamics are a separate check (stability under gravity, no interpenetration). Articulate-Anything's render-and-compare critic is the closest published analogue.

### Cited Findings
- Articulate-Anything uses a closed loop: generate the URDF, render or simulate it, and have a VLM critic compare the result against the input video. The critic tends to report success too readily. — [Articulate-Anything](https://www.seas.upenn.edu/~eeaton/papers/Le2025ArticulateAnything.pdf)
- Joint limits, armature and positive-definite inertia are prerequisites for stable MuJoCo playback. — [MuJoCo XMLreference.rst](https://raw.githubusercontent.com/google-deepmind/mujoco/main/doc/XMLreference.rst)

### Inferences
- **Suggested validation step**:
  - **(a) Compile check**: run `mujoco.MjModel.from_xml_path` with no `boundmass` or `boundinertia` so that invalid inertia fails loudly, and require all masses > 0 and all limits set.
  - **(b) FK replay**: for each frame t, set `data.qpos = q̂_t` and call `mj_kinematics`. Compute the per-part pose error against observed part poses (rotation geodesic in degrees, translation in metres) and the 2D mask IoU or Chamfer distance against observed points.
  - **(c) Range check**: confirm the observed states fall within [lower, upper].
  - **(d) Passive dynamics**: step about 2 s with zero control and confirm no explosion (finite qpos, energy bounded) and no persistent contact penetration at the rest pose.
  - **(e) Reachability**: sweep each joint across its range and confirm no self-collision with the static body.
- **What FK replay cannot show**: it validates geometry and kinematics only, and says nothing about mass, damping or friction. Those need force or interaction data.

### Gaps
- I found no paper that formally defines a "sim replay trajectory error" metric. The protocol above is my recommendation and is not sourced.
